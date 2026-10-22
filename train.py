import os
import torch
import random
import numpy as np
import time
import argparse
import json
import csv
import sys

from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model
import dataloader

MODEL_DIR = None
LOG_DIR = None


class Logger():
	def __init__(self,path):
		self.path = path
		self.header = None

	def log(self,metrics):

		if self.header is None:
			self.header = list(metrics.keys())
			with open(self.path,'w') as fp:
				fp.write('\t'.join(self.header)+'\n')

		row = [f'{v:.5f}' if isinstance(v,float) else str(v) for v in metrics.values()]
		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


class RecentBestTracker():
	'''
	Keep only 'n' .pth.tar files.
	'''
	def __init__(self,n=3):
		self.n = n
		self.paths = [] #FIFO for top n models.

	def update(self,path):
		self.paths.append(path)
		if len(self.paths) > self.n:
			old_path = self.paths.pop(0)
			if os.path.exists(old_path):
				os.remove(old_path)


def save_checkpoint(path,model,optim,epoch,metrics,tag):
	save_path = f"{path}/{tag}_{model.model_id:03}_e{epoch:04}.pth.tar"

	raw_model = model._orig_mod if hasattr(model,'_orig_mod') else model

	chkpt = {
		'epoch': epoch,
		'tloss': metrics['tloss'],
		'v_err': metrics['rmse'],
		'model_state_dict': raw_model.state_dict(),
		'optim_state_dict': optim.state_dict()
	}

	torch.save(chkpt,save_path)
	return save_path


def set_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def embedding_std(emb):
	'''
	Mean per-dimension std of l2-normalized embeddings, scaled by sqrt(D).
	~1 for well-spread embeddings, ->0 under collapse.
	'''
	# emb is [B,D]
	emb = torch.nn.functional.normalize(emb.float(),dim=1) #unit-length vectors
	return (emb.std(dim=0).mean() * emb.shape[1]**0.5) # no SYNC need .item() downstream


def unpermute2d(representation,patch_ids):
	'''
	Undo flip/rotation augmentations done to input.
	Returns corrected feature map from model output.

	representation: [B,C,G,G] encoder output
	patch_ids:      [B,G,G] patch_ids[b,i,j] original patch of cell (i,j)
	'''
	B,C = representation.shape[:2]
	f   = representation.flatten(2).transpose(1,2)
	inv = patch_ids.flatten(1).argsort(dim=1)
	return f.gather(1,inv[...,None].expand(-1,-1,C))


def unpermute1d(emb,patch_ids):
	inv = patch_ids.flatten(1).argsort(dim=1)
	return emb.gather(1,inv[...,None].expand(-1,-1,emb.shape[-1]))


def load_tract_values(split_tracts_csv,values_csv,n_tracts=0):
	'''
	Array indexed by tract id (tract_index) holding the health value, joined on GEOID.
	NaN for ids without a value. At least n_tracts long so any tract id from the chips is a valid index.
	'''

	# LOAD {GEOID:diabetes value}
	with open(values_csv) as fp:
		by_geoid = {r['LocationID']: float(r['Data_Value']) for r in csv.DictReader(fp)}

	# LOAD GEOID + INDEX IN ORIGINAL (VALUES CSV)
	with open(split_tracts_csv) as fp:
		tracts = list(csv.DictReader(fp))

	size = max(max(int(r['tract_index']) for r in tracts)+1,n_tracts)
	values = np.full(size,np.nan,dtype=np.float32)

	for r in tracts:
		values[int(r['tract_index'])] = by_geoid.get(r['GEOID'],np.nan)
	return values


@torch.no_grad()
def build_table(model,dataloader,device):

	embed_dim = model.representation_dim
	n_tracts  = dataloader.dataset.n_tracts

	# THE *_ftr.tif TRACT BAND STORES tract_index+1, KEEPING 0 FREE AS NODATA, SO
	# BUCKETS RUN 0..n_tracts AND BUCKET 0 IS DISCARDED BELOW.
	n_buckets = n_tracts + 1

	# BUFFERS
	embeddings = torch.zeros((n_buckets,embed_dim),device=device)
	counts     = torch.zeros(n_buckets,dtype=torch.long,device=device)

	model.eval()
	for image,tract in dataloader:

		image = image.to(device,non_blocking=True)
		tract = tract.to(device,non_blocking=True)

		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
			rep = model(image)

		# PIXELS PER (CHIP,BUCKET): [B,H*W] -> [B,n_buckets]
		B   = tract.shape[0]
		t   = tract.view(B,-1).long()
		off = torch.arange(B,device=device)[:,None]*n_buckets
		W   = torch.bincount((t+off).flatten(),minlength=B*n_buckets).view(B,n_buckets)

		# PIXEL-WEIGHTED SUM OF CHIP EMBEDDINGS PER TRACT
		embeddings += W.T.float() @ rep.float() #[n_buckets,B] x [B,embed_dim]
		counts     += W.sum(0)

	# KEEP SEEN ROWS, DROPPING THE NODATA BUCKET + AVERAGE
	ids = (counts > 0).nonzero().squeeze(1)
	ids = ids[ids > 0]
	tract_embedding_table = embeddings[ids] / counts[ids,None]

	# SPREAD OF THE TABLE THE PROBE WILL FIT: ~1 WELL-SPREAD, ->0 UNDER COLLAPSE.
	# PER-TRACT AND CHIP-AVERAGED, SO NOT ON THE SAME SCALE AS tstd FROM train().
	std = embedding_std(tract_embedding_table).item()

	# BACK TO 0-BASED tract_index, SO THE CALLER INDEXES load_tract_values() DIRECTLY
	return (ids - 1).cpu().numpy(),tract_embedding_table.cpu().numpy(),std


def score_probe(X_tr,X_va,Y_tr,Y_va):
	'''
	Calculate the regression on health variable from the avg tract embedding table.
	'''
	# DROP TRACTS WITHOUT A HEALTH VALUE
	m_tr,m_va = ~np.isnan(Y_tr),~np.isnan(Y_va)
	X_tr,Y_tr = X_tr[m_tr],Y_tr[m_tr]
	X_va,Y_va = X_va[m_va],Y_va[m_va]

	# THE TABLE COMES IN AS float32 AND sklearn KEEPS THAT DTYPE ALL THE WAY INTO THE
	# CHOLESKY SOLVE OF (X^T X + alpha I). EMBEDDING DIMS ARE HIGHLY CORRELATED AND THE
	# SMALLEST alpha IS 1, SO cond(A) ~1e7 -- BELOW float32 eps (1.2e-7), WHICH IS WHAT
	# TRIGGERS LinAlgWarning AND MAKES THE FITTED COEFFICIENTS (NOT SO MUCH THE
	# PREDICTIONS) UNRELIABLE. float64 COPY IS ~200MB/CHEAP.
	X_tr,X_va = X_tr.astype(np.float64),X_va.astype(np.float64)
	Y_tr,Y_va = Y_tr.astype(np.float64),Y_va.astype(np.float64)

	# DEFAULT
	alphas = np.logspace(start=0,stop=6,num=8)
	folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X_tr))

	# FIT RIDGE REGRESSION TO TRAIN SET EMBEDDINGS
	probe = make_pipeline(StandardScaler(),RidgeCV(alphas=alphas,cv=folds)).fit(X_tr, Y_tr)

	# PREDICT VALIDATION HEALTH VALUES FROM VALIDATION EMBEDDINGS
	y_hat = probe.predict(X_va)
	err = y_hat - Y_va
	alpha = probe[-1].alpha_

	scores = {
		"rmse": float(np.sqrt(np.mean(err ** 2))),
		"mae": float(np.mean(np.abs(err))),
		"r2": float(1 - np.mean(err ** 2) / np.var(Y_va)),  # 0 = no better than predicting the mean
		"spearman": float(spearmanr(Y_va, y_hat)[0]),       # are tracts ranked correctly?
		"alpha": float(alpha),
		"alpha_edge": bool(np.isclose(alpha, alphas[0]) or np.isclose(alpha, alphas[-1])),  # True -> widen alphas
	}
	return scores, y_hat, probe


def train(model,loss_fn,dataloaders,optimizer,scheduler,device):

	sums = torch.zeros(4,device=device)
	batch_nr = torch.zeros(1,device=device)

	model.train()
	for X1,X2 in dataloaders['training']: # (X1,ids1),(X2,ids2) for patched+rotated

		#TO DEV
		X1 = X1.to(device,non_blocking=True)
		X2 = X2.to(device,non_blocking=True)

		# FORWARD PASS
		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
			rep,z1,z2 = model(X1,X2)

        # z1 = unpermute1d(z1,ids1).flatten(0,1) # [B*N,D], row r = same (image,patch)
        # z2 = unpermute1d(z2,ids2).flatten(0,1)

		# LOSS (on device, in float32)
		loss,on_diag,off_diag = loss_fn(z1,z2)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		std = embedding_std(rep.mean(dim=(2,3))) #spread of embeddings, ~1 for well-spread, ->0 under collapse

		sums += torch.stack([loss.detach(),on_diag,off_diag,std])
		batch_nr += 1

	if scheduler is not None:
		scheduler.step() #step per epoch

	loss,diag,offd,std = (sums/batch_nr).tolist() # --- GPU-CPU SYNC ---
	print(f'[T] LOSS: {loss:.5f} | ON_DIAG: {diag:.5f} | OFF_DIAG: {offd:.5f} | STD: {std:.5f}')
	return {'tloss':loss,'tdiag':diag,'toffd':offd,'tstd':std}


def train_and_validate(model,loss_fn,dataloaders,optimizer,scheduler,epochs,device):

	# LOGGING
	logger = Logger(f'{LOG_DIR}/epochs_{model.model_id:03}.tsv')
	chkpt_tracker = RecentBestTracker(n=3)
	best_epoch = 0
	best_rmse = float('inf')

	# HEALTH VALUE PER TRACT ID (Y = tract_values[ids] AFTER build_table)
	n_tracts = dataloaders['training_labeled'].dataset.n_tracts
	tract_values = load_tract_values('./split_tracts.csv','./filtered_diabetes.csv',n_tracts)

	# TRAIN AND VALIDATE -- RUN
	overall_t0 = time.perf_counter()
	for epoch in range(epochs):

		print(f'--- EPOCH {epoch:04} ---')
		epoch_time_t0 = time.perf_counter()

		# TRAIN
		train_time_t0 = time.perf_counter()
		tr_result = train(model,loss_fn,dataloaders,optimizer,scheduler,device)
		train_time_t1 = time.perf_counter()

		# EVERY 10 EPOCHS -- VALIDATE: BUILD PER-TRACT EMBEDDINGS, FIT PROBE, PREDICT DIABETES
		is_eval_epoch = (epoch+1) % 10 == 0
		if is_eval_epoch:

			# BUILD TRAIN + VALIDATION TABLES
			ids_t,avg_emb_t,std_t = build_table(model,dataloaders['training_labeled'],device)
			ids_v,avg_emb_v,std_v = build_table(model,dataloaders['validation'],device)

			# HEALTH VALUES ALIGNED TO TABLE ROWS
			Y_tr = tract_values[ids_t]
			Y_va = tract_values[ids_v]

			# FIT REGRESSION
			scores,_,_ = score_probe(avg_emb_t,avg_emb_v,Y_tr,Y_va)
			scores['std_tr'] = std_t
			scores['std_va'] = std_v
			print(f"[V] RMSE: {scores['rmse']:.5f} | R2: {scores['r2']:.5f} | SPEARMAN: {scores['spearman']:.5f}"
			      f" | STD_TR: {std_t:.5f} | STD_VA: {std_v:.5f}")

		else:
			scores = {
				"rmse":'NaN',
				"mae": 'NaN',
				"r2": 'NaN',
				"spearman": 'NaN',
				"alpha": 'NaN',
				"alpha_edge": 'NaN',
				"std_tr": 'NaN',
				"std_va": 'NaN'
			}

		# LOG EPOCH
		epoch_results = {'epoch':epoch,**tr_result,**scores}
		logger.log(epoch_results)

		# SAVE CHECKPOINT
		if is_eval_epoch and scores['rmse'] < best_rmse:
			best_rmse = scores['rmse']
			best_epoch = epoch
			path = save_checkpoint(MODEL_DIR,model,optimizer,epoch,epoch_results,tag='best')
			chkpt_tracker.update(path)

		epoch_time_t1 = time.perf_counter()
		print(f"Train time: {train_time_t1-train_time_t0:.2f} secs.")
		print(f"Epoch time: {epoch_time_t1-epoch_time_t0:.2f} secs.")

	# STDOUT TIME/OVERALL BEST/VRAM
	overall_t1 = time.perf_counter()
	print(f'Total time: {(overall_t1-overall_t0)/60:.1f} min')
	print(f'Best epoch: {best_epoch} | rmse: {best_rmse:.5f}')
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')


def load_hyperparameters(args):

	# LOAD FILE
	with open(args.params,'r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line.strip()]
	assert len(hp_list) > 0, f"Got empty file for {args.params}"

	# SET IDs AS KEYS and CHECK
	hp_list_indexed = {row['id']:row for row in hp_list}
	assert args.id in hp_list_indexed, f"model id '{args.id}' not in hyperparameter file {args.params}"

	# SET DICT FOR CURRENT MODEL
	HP = hp_list_indexed[args.id]

	# CHECK HERE -- TODO TRY-EXCEPT TO SPECIFIC ERR
	pass

	return HP


def parse_args():
	# SET + LOAD
	parser = argparse.ArgumentParser(description='Barlow Twins pre-training on Sentinel-2 chips.')
	parser.add_argument('--id',type=int,default=0,help='Model key id')
	parser.add_argument('--data-dir',type=str,required=True,help='Dir for chips.')
	# parser.add_argument('--mask-dir',type=str,required=True,help='Dir for feature/tract masks (*_ftr.tif).')
	parser.add_argument('--log-dir',type=str,required=True,help='Dir for logs outputted by train.py')
	parser.add_argument('--checkpoint-dir',type=str,required=True,help='Dir for ouputted checkpoint weights.')
	parser.add_argument('--device',required=False,type=int,default=0,help='GPU index')
	parser.add_argument('--num-workers',required=False,type=int,default=8)
	parser.add_argument('-p','--params',required=True,help='JSON-lines hyperparameter file, e.g. ./hparams/hpo_1.json')
	args = parser.parse_args()

	assert os.path.isdir(args.data_dir), f"Data dir not found at {args.data_dir}"
	# assert os.path.isdir(args.mask_dir), f"Mask dir not found at {args.mask_dir}"
	assert os.path.isdir(args.log_dir), f"Log dir not found at {args.log_dir}"
	assert os.path.isdir(args.checkpoint_dir), f"Checkpoint dir not found at {args.checkpoint_dir}"
	assert os.path.isfile(args.params), f"Hyperparameter dict path not found at {args.params}"

	assert args.device >= 0, "Incorrect index in GPU id"
	assert args.device < torch.cuda.device_count(), "Out of range GPU id"

	global MODEL_DIR
	global LOG_DIR
	MODEL_DIR = args.checkpoint_dir
	LOG_DIR   = args.log_dir

	return args


if __name__ == '__main__':
	args = parse_args()

	# HYPERPARAMETERS
	hp = load_hyperparameters(args)
	epochs = hp['epochs']

	# SET SEED
	set_seed(hp['seed'])

	# SET GPU
	device = torch.device(f'cuda:{args.device}')

	# LOAD MODEL
	bt_net = model.SentinelBarlowTwins(
		model_id=hp['id'],
		bands=hp['bands'],
		init_chan=hp['init_chan'],
		cnn_layers=hp['cnn_layers'],
		proj_dim=hp['proj_dim']
	).to(device)

	bt_net = torch.compile(bt_net)

	# LOSS
	loss_fn = model.BarlowTwinsLoss(proj_dim=hp['proj_dim'],lambd=hp['bt_lambda']).to(device)

	# OPTIMIZER
	optimizer = torch.optim.AdamW(bt_net.parameters(),lr=hp['lrate'],weight_decay=hp['decay'])

	# # RAM DATASETS+DATA (~37GB)
	print(f"Loading cached datasets to RAM from .npy files in {args.data_dir}...")
	t0_disk_to_ram = time.perf_counter()
	tr_bands,tr_tracts = dataloader.load_cached_split(args.data_dir,split='train')
	va_bands,va_tracts = dataloader.load_cached_split(args.data_dir,split='val')
	t1_disk_to_ram = time.perf_counter()
	print("Loaded to RAM! -- ",end='')
	print(f"{t1_disk_to_ram-t0_disk_to_ram:.2f} secs.")

	# PRINT RAM USAGE -- TODO: PSUTIL NOT INSTALLED IN THE K8S IMAGE
	# pid = os.getpid()
	# python_process = psutil.Process(pid)
	# mem_bytes = python_process.memory_info().rss
	# mem_gb = mem_bytes / (1024 ** 3)


	train_transform = dataloader.ImageTransform()

	# DATALOADERS
	loader_params = {
		'batch_size': hp['batch'],
		'num_workers': args.num_workers,
		'pin_memory': True,
		'prefetch_factor': 4,
		'persistent_workers': True
	}

	# PAIRED + AUGMENTED -- BARLOW TWINS TRAINING
	t_dload = torch.utils.data.DataLoader(
		dataset=dataloader.RAMPairedDataset(tr_bands,transform=train_transform,split='train'),
		shuffle=True,
		drop_last=True,
		**loader_params
	)

	# LABELED, UNAUGMENTED -- TRAIN TABLE FOR THE PROBE
	t_dload_labeled = torch.utils.data.DataLoader(
		dataset=dataloader.RAMLabeledDataset(tr_bands,tr_tracts,split='train'),
		shuffle=False,
		drop_last=False,
		**loader_params
	)

	v_dload = torch.utils.data.DataLoader(
		dataset=dataloader.RAMLabeledDataset(va_bands,va_tracts,split='val'),
		shuffle=False,
		drop_last=False,
		**loader_params
	)

	dataloaders = {'training': t_dload,'training_labeled': t_dload_labeled,'validation': v_dload}

	# SCHEDULER -- COSINE ANNEALING W/O RESTARTS
	warmup_steps = 10
	cosine_steps = epochs-warmup_steps
	min_rate = hp['lrate'] / cosine_steps
	warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cosine_steps,eta_min=min_rate)
	scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

	# TRAIN AND VALIDATE
	train_and_validate(bt_net,loss_fn,dataloaders,optimizer,scheduler,epochs,device)
