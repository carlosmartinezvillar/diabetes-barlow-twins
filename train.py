import os
import torch
import random
import numpy as np
import time
import argparse
import json
import csv

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model
import dataloader

MODEL_DIR = None
MASK_DIR = None
DATA_DIR = None
LOG_DIR = None
EMBED_DIM = None


class Logger():
	def __init__(self,path):
		self.path = path
		self.header = None

	def log(self,metrics):
		# row = [f'{v:.6f}' if isinstance(v,float) else str(v) for v in metrics.values()]
		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


def save_checkpoint(path,model,optim,epoch,metrics,tag):
	pass


def set_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


@torch.no_grad()
def embedding_std(emb):
	'''
	Mean per-dimension std of l2-normalized embeddings, scaled by sqrt(D).
	~1 for well-spread embeddings, ->0 under collapse.
	'''
	# emb is [B,D]
	emb = torch.nn.functional.normalize(emb.float(),dim=1) #unit-length vectors
	return (emb.std(dim=0).mean() * emb.shape[1]**0.5).item() # SYNC


def score_probe(X_tr,X_va,Y_tr,Y_va):
	'''
	Calculate the regression on health variable from the avg tract embedding table.
	'''
	# DEFAULT
	alphas = np.logspace(0, 6, 8)
	folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X_tr))

	# FIT RIDGE TO TRAIN SET
	probe = make_pipeline(StandardScaler(),RidgeCV(alphas=alphas,cv=folds)).fit(X_tr, Y_tr)

	# PREDICT VAL VALUES
	y_hat = probe.predict(X_va)
	error = y_hat - Y_va
	alpha = probe[-1].alpha_

	scores = {
		"rmse": float(np.sqrt(np.mean(err ** 2))),
		"mae": float(np.mean(np.abs(err))),
		"r2": float(1 - np.mean(err ** 2) / np.var(y_va)),  # 0 = no better than predicting the mean
		"spearman": float(spearmanr(y_va, y_hat)[0]),       # are tracts ranked correctly?
		"alpha": float(alpha),
		"alpha_at_edge": bool(np.isclose(alpha, alphas[0]) or np.isclose(alpha, alphas[-1])),  # True -> widen alphas
	}
	return scores, y_hat, probe


def load_tract_values(split_tracts_csv,values_csv):
	'''
	Array indexed by tract id (tract_index) holding the health value, joined on GEOID.
	NaN for ids without a value.
	'''
	with open(values_csv) as fp:
		by_geoid = {r['LocationID']: float(r['Data_Value']) for r in csv.DictReader(fp)}

	with open(split_tracts_csv) as fp:
		tracts = list(csv.DictReader(fp))

	values = np.full(max(int(r['tract_index']) for r in tracts)+1,np.nan,dtype=np.float32)

	for r in tracts:
		values[int(r['tract_index'])] = by_geoid.get(r['GEOID'],np.nan)
	return values


@torch.no_grad()
def build_table(model,dataloader):

	embed_dim = model.representation_dim
	n_tracts  = dataloader.n_tracts

	# BUFFERS
	embeddings = torch.zeros((n_tracts,embed_dim),device=device)
	counts     = torch.zeros(n_tracts,dtype=torch.long,device=device)

	for image,label,tract in dataloader:

		image = image.to(device,non_blocking=True)
		tract = tract.to(device,non_blocking=True)

		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
			rep = model(image)

		# PIXELS PER (CHIP,TRACT): [B,H*W] -> [B,n_tracts]
		B   = tract.shape[0]
		t   = tract.view(B,-1).long()
		off = torch.arange(B,device=device)[:,None]*n_tracts
		W   = torch.bincount((t+off).flatten(),minlength=B*n_tracts).view(B,n_tracts)

		# PIXEL-WEIGHTED SUM OF CHIP EMBEDDINGS PER TRACT
		embeddings += W.T.float() @ rep.float() #[n_tracts,B] x [B,embed_dim]
		counts     += W.sum(0)

	# KEEP TRACT IDS OF SEEN ROWS + AVERAGE
	ids = (counts > 0).nonzero().squeeze(1)
	tract_embedding_table = embeddings[ids] / counts[ids,None]

	return ids.cpu().numpy(),tract_embedding_table.cpu().numpy()


def train(model,loss_fn,dataloaders,optimizer,scheduler,device):

	sums = torch.zeros(3,device=device)
	batch_nr = torch.zeros(1,device=device)

	model.train()
	for X1,X2 in dataloaders['training']:

		#TO DEV
		X1 = X1.to(device,non_blocking=True)
		X2 = X2.to(device,non_blocking=True)

		# FORWARD PASS
		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
			_,z1,z2 = model(X1,X2)

		# LOSS (on device, in float32)
		loss,on_diag,off_diag = loss_fn(z1,z2)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		sums += torch.stack([loss.detach(),on_diag,off_diag])
		batch_nr += 1

	if scheduler is not None:
		scheduler.step() #step per epoch

	loss,diag,offd = (sums/batch_nr).tolist() # --- GPU-CPU SYNC ---
	print(f'[T] LOSS: {loss:.5f} | ON_DIAG: {diag:.5f} | OFF_DIAG: {offd:.5f}')
	return {'tloss':loss,'tdiag':diag,'toffd':offd}


def validate(model,loss_fn,dataloaders,device,n_tracts,embed_dim):
	sums     = torch.zeros(3,device=device)
	batch_nr = torch.zeros(1,device=device)
	embeddings = torch.zeros((embed_dim),device=device)

	model.eval()
	with torch.no_grad():
		for X1,X2 in dataloaders['validation_paired']:

			#TO DEV
			X1 = X1.to(device,non_blocking=True)
			X2 = X2.to(device,non_blocking=True)

			#FORWARD
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
				_,z1,z2 = model(image)
			batch_nr += 1

	#stddev of (image-level) embeddings  -- eval collapse
	std = embedding_std(torch.cat(embeddings))
	# print(f'[V] LOSS: {loss:.5f} | ON_DIAG: {diag:.5f} | OFF_DIAG: {offd:.5f} | EMB_STD: {std:.4f}')
	# return {'vloss':loss,'vdiag':diag,'voffd':offd,'vstd':std}


def train_and_validate(model,loss_fn,dataloaders,optimizer,scheduler,epochs,device):

	# LOGGING
	logger = Logger(f'{LOG_DIR}/epochs_{model.model_id:03}.tsv')
	best_epoch = 0
	# best_vloss = float('inf')


	# HEALTH VALUE PER TRACT ID (Y = tract_values[ids] AFTER build_table)
	tract_values = load_tract_values('./split_tracts.csv','./filtered_diabetes.csv')

	overall_t0 = time.perf_counter()
	for epoch in range(epochs):

		print(f'--- EPOCH {e:03} ---')
		epoch_t0 = time.perf_counter()

		# TRAIN
		tr_result = train(model,loss_fn,dataloaders,optimizer,scheduler,device)

		# AT EPOCH K -- VALIDATE: BUILD TRACT-EMBEDDING TABLES, FIT PROBE, EVALUATE
		if epoch+1 > 10 and (epoch+1) % 10 == 0:

			# BUILD TRAIN TABLE
			model.eval()
			ids_t,avg_emb_t = build_table(model,dataloaders['training'])

			# BUILD VALIDATION TABLE
			model.eval()
			ids_v,avg_emb_v = build_table(model,dataloaders['validation'])

			# HEALTH VALUES ALIGNED TO TABLE ROWS
			Y_tr = tract_values[ids_t]
			Y_va = tract_values[ids_v]

			# FIT REGRESSION
			scores = score_probe(avg_emb_t,avg_emb_v,Y_tr,Y_va)

		epoch_t1 = time.perf_counter()
		print(f"Epoch time: {epoch_t1-epoch_t0:.2f} secs.")

		# LOG EPOCH
		# epoch_results = calculate_metrics(tr_result,va_result,optimizer)
		logger.log(epoch_results)

		# SAVE CHECKPOINT
		# save_checkpoint(tag,model,optimizer,epoch)
		if va_result['vloss'] < best_vloss:
			best_vloss = va_result['vloss']
			best_epoch = epoch
			save_checkpoint(MODEL_DIR, model,optimizer,epoch)


	overall_t1 = time.perf_counter()

	# STDOUT TIME/OVERALL BEST/VRAM
	print(f'Total time: {(overall_t1-overall_t0)/60:.1f} min')
	print(f'Best epoch: {best_epoch} | vloss: {best_vloss:.5f}')
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')


def load_hyperparameters(args):
	pass


def parse_args():
	parser = argparse.ArgumentParser(description='Barlow Twins pre-training on Sentinel-2 chips.')
	parser.add_argument('--id',type=int,default=0,help='Model key id')
	parser.add_argument('--data-dir',type=str,default=None,help='Dir for chips.')
	parser.add_argument('--log-dir',type=str,default=None)
	parser.add_argument('--checkpoint-dir',type=str,default=None)
	parser.add_argument('--device',type=str,default='0')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()

	# HYPERPARAMETERS
	hp = load_hyperparameters(args.id)

	# SET SEED
	set_seed(hp['seed'])

	# SET GPU
	device = torch.device(f'cuda:{args.device}')

	# LOAD MODEL
	bt_net = model.SentinelBarlowTwins(
		model_id=hp['model_id'],
		bands=hp['bands'],
		init_chan=hp['init_chan'],
		cnn_layers=hp['cnn_layers'],
		proj_dim=hp['proj_dim']
	).to(device)
	# bt_net = model.SentinelBarlowTwins(**hp)

	bt_net = torch.compile(bt_net)

	# LOSS
	loss_fn = model.BarlowTwinsLoss(proj_dim=hp['proj_dim'],lambd=hp['bt_lambda']).to(device)

	# OPTIMIZER
	optimizer = torch.optim.AdamW(bt_net.parameters(),lr=hp['lrate'],weight_decay=hp['decay'])

	# DATALOADERS
	loader_params = {
		'batch_size': hp['batch'],
		'num_workers': args.num_workers,
		'prefetch_factor':4,
		'persistent_workers':True
	}

	train_transform = dataloader.ImageTransform()

	t_dload = torch.utils.data.DataLoader(
		dataset=dataloader.SentinelPairedDataset(args.data_dir,transform=train_transform,split='train'),
		shuffle=True,
		drop_last=True,
		**loader_params
	)

	v_dload = torch.utils.data.DataLoader(
		dataset=dataloader.SentinelLabeledDataset(args.data_dir,args.mask_dir,split='val'),
		shuffle=False,
		drop_last=False,
		**loader_params
	)

	dataloaders = {'training': t_dload,'validation': v_dload}

	# SCHEDULER -- COSINE ANNEALING 1 CYCLE/NO RESTARTS
	# scheduler = None

	# TRAIN AND VALIDATE
	train_and_validate(bt_net,loss_fn,dataloaders,optimizer,scheduler,epochs,device)




