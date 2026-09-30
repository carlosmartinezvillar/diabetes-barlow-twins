import os
import torch
import random
import numpy as np
import time
import argparse
import json

import model
import dataloader


class FileLogger():
	def __init__(self,path):
		self.path = path

	def log(self,metrics):
		row = ["",""]
		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


def calculate_metrics():
	pass

def save_checkpoint():
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


def train(model,dataloaders,optimizer,device):

	sums = torch.zeros(3,device=device)
	batch_nr = torch.zeros(1,device=device)

	autocast=True

	model.train()
	for X1,X2 in dataloaders['training']:

		#TO DEV
		X1 = X1.to(device,non_blocking=True)
		X2 = X2.to(device,non_blocking=True)

		# FORWARD PASS
		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=autocast):
			loss,on_diag,off_diag = model(X1,X2)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		# scheduler.step()

		sums += torch.stack([loss.detach(),on_diag,off_diag])
		batch_nr += 1

	loss,diag,offd = (sums/batch_nr).tolist() # --- GPU-CPU SYNC ---
	print(f'[T] LOSS: {loss:.5f} | ON_DIAG: {diag:.5f} | OFF_DIAG: {offd:.5f}')
	return {'tloss':loss,'tdiag':diag,'toffd':offd}

def validate(model,dataloaders):
	sums = torch.zeros(3,device=device)
	batch_nr = torch.zeros(1,device=device)

	model.eval()
	with torch.no_grad():
		for X1,X2 in dataloaders['validation']:
			#TO DEV
			X1 = X1.to(device,non_blocking=True)
			X2 = X2.to(device,non_blocking=True)

			#FORWARD
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=autocast):
				loss,on_diag,off_diag = model(X1,X2)
				# embeddings.append(raw_model.embed(X1).float())

			sums += torch.stack([loss.detach(),on_diag,off_diag])
			batch_nr += 1				

	loss,diag,offd = (sums/batch_nr).tolist() # --- GPU-CPU SYNC ---

	#stddev of embeddings?
	# embedding_std()

	return {'vloss':loss,'vdiag':diag,'voffd':offd}


def train_and_validate(model,dataloaders,optimizer,scheduler,epochs):

	logger = FileLogger(f'{LOG_DIR}/training_{model.model_id:03}.tsv')
	best_epoch = 0

	for e in range(epochs):

		# TRAIN
		tr_result = train(model,dataloaders,optimizer,scheduler)

		# VALIDATE
		va_result = validate(model,dataloaders)

		# LOG EPOCH
		logger.log(epoch_results)

		# SAVE CHECKPOINT

	# STDOUT TIME/OVERALL BEST/VRAM
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')

def load_hyperparameters():

def parse_args():

if __name__ == '__main__':
	pass
	# parse_args()

	# HYPERPARAMETERS

	# SET SEED

	# LOAD MODEL

	# OPTIMIZER

	# DATALOADERS

	# SCHEDULER -- LATER

	# TRAIN AND VALIDATE
	train_and_validate(bt_net,dataloaders,optimizer,scheduler=None,epochs=hp['epochs'])
	