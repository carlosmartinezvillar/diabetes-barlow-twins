import os
import torch
import random
import numpy as np
import time
import argparse
import json

import model
import dataloader

MODEL_DIR = None
MASK_DIR = None
DATA_DIR = None
LOG_DIR = None
EMBED_DIM = 1024


class Logger():
	def __init__(self,path):
		self.path = path

	def log(self,metrics):
		# row = [f'{v:.6f}' if isinstance(v,float) else str(v) for v in metrics.values()]
		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


@torch.no_grad()
def calculate_metrics(pred,target):
	pass


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
	emb = torch.nn.functional.normalize(emb.float(),dim=1)
	return (emb.std(dim=0).mean() * emb.shape[1]**0.5).item()


def train(model,loss_fn,dataloaders,optimizer,scheduler,device,autocast=True):

	sums = torch.zeros(3,device=device)
	batch_nr = torch.zeros(1,device=device)

	model.train()
	for batch in dataloaders['training']:
		X1,X2 = batch[0],batch[1]

		#TO DEV
		X1 = X1.to(device,non_blocking=True)
		X2 = X2.to(device,non_blocking=True)

		# FORWARD PASS
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=autocast):
			_,z1,z2 = model(X1,X2)

		# LOSS (on device, in float32)
		loss,on_diag,off_diag = loss_fn(z1,z2)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		if scheduler is not None:
			scheduler.step()

		sums += torch.stack([loss.detach(),on_diag,off_diag])
		batch_nr += 1

	loss,diag,offd = (sums/batch_nr).tolist() # --- GPU-CPU SYNC ---
	print(f'[T] LOSS: {loss:.5f} | ON_DIAG: {diag:.5f} | OFF_DIAG: {offd:.5f}')
	return {'tloss':loss,'tdiag':diag,'toffd':offd}


def validate(model,loss_fn,dataloaders,device,autocast=True):
	sums     = torch.zeros(3,device=device)
	batch_nr = torch.zeros(1,device=device)
	embeddings = torch.zeros((n_tracts,EMBED_DIM))

	model.eval()
	with torch.no_grad():
		for X1,X2 in dataloaders['validation']:

			#TO DEV
			X1 = X1.to(device,non_blocking=True)
			X2 = X2.to(device,non_blocking=True)

			#FORWARD
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=autocast):
				emb,z1,z2 = model(X1,X2)

			loss,on_diag,off_diag = loss_fn(z1,z2)
			embeddings.append(emb.mean(dim=1).float()) # [B,N,D] -> [B,D]

			sums += torch.stack([loss.detach(),on_diag,off_diag])
			batch_nr += 1

	loss,diag,offd = (sums/batch_nr).tolist() # --- GPU-CPU SYNC ---

	#stddev of (image-level) embeddings
	std = embedding_std(torch.cat(embeddings))

	print(f'[V] LOSS: {loss:.5f} | ON_DIAG: {diag:.5f} | OFF_DIAG: {offd:.5f} | EMB_STD: {std:.4f}')
	return {'vloss':loss,'vdiag':diag,'voffd':offd,'vstd':std}


def train_and_validate(model,loss_fn,dataloaders,optimizer,scheduler,epochs,device):

	logger = Logger(f'{LOG_DIR}/training_{model.model_id:03}.tsv')
	best_epoch = 0
	best_vloss = float('inf')

	overall_t0 = time.perf_counter()

	for epoch in range(epochs):
		print(f'--- EPOCH {e:03} ---')
		epoch_t0 = time.perf_counter()

		# TRAIN
		tr_result = train(model,loss_fn,dataloaders,optimizer,scheduler,device)

		# VALIDATE
		va_result = validate(model,loss_fn,dataloaders,device)

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
		init_cha/n=hp['init_chan'],
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

	train_transform = dataloader.BarlowTwinsTransform()

	t_dload_1 = torch.utils.data.DataLoader(
		dataset=dataloader.SentinelPairDataset(f'{args.data_dir}/training',train_transform),
		shuffle=True,
		drop_last=True,
		**loader_params
	)

	t_dload_2 = torch.utils.data.DataLoader(
		dataset=dataloader.SentinelDataset(f'{args.data_dir}/training',f'{args.mask_dir}'),
		shuffle=False,
		drop_last=False,
		**loader_params
	)

	v_dload = torch.utils.data.DataLoader(
		dataset=dataloader.SentinelDataset(f'{args.data_dir}/validation',f'{args.mask_dir}'),
		shuffle=False,
		drop_last=False,
		**loader_params
	)
	dataloaders = {'training': t_dload_1,'training_labeled': t_load_2,'validation': v_dload}

	# SCHEDULER -- LATER
	# scheduler = None

	# TRAIN AND VALIDATE
	train_and_validate(bt_net,loss_fn,dataloaders,optimizer,scheduler,epochs,device)
