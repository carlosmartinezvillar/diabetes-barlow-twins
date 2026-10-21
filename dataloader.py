import torch
import torchvision
from torchvision import tv_tensors
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as v1
import torchvision.transforms.v2 as v2
import glob
import os
import tifffile as tiff
import random
import csv
import argparse

# TRAINING SUBSET MEAN AND STD DEV.
MEAN = torch.tensor([83.8274,78.5392,83.9449,77.0361,97.3247,
	146.737,141.844,147.537,126.045,97.1117]).view(-1,1,1)

STDD = torch.tensor([59.2182,61.9259,58.5977,65.5007,57.8101,
		50.3012,52.5063,51.4996,57.7896,66.0391]).view(-1,1,1)

################################################################################
# HELPERS
################################################################################
def read_split(split_csv,split):
	'''
	Returns the split's chip ids and n_tracts = max tract id + 1 over ALL splits,
	since tract ids are global and used directly as row indices.
	'''
	with open(split_csv) as fp:
		rows = list(csv.DictReader(fp))
	chips    = sorted(r['chip'] for r in rows if r['split']==split)
	n_tracts = max(int(t) for r in rows for t in r['tracts'].split()) + 1
	return chips,n_tracts


def build_cached_dataset(chip_dir,feat_dir,split_csv,split,cache_dir):
	chips,_ = read_split(split_csv,split)
	n     = len(chips)
	os.makedirs(cache_dir,exist_ok=True)
	rgb = np.lib.format.open_memmap(f"{cache_dir}/{split}_rgb.npy",mode='w+',dtype=np.uint8,shape=(n,10,256,256))
	lbl = np.lib.format.open_memmap(f"{cache_dir}/{split}_lbl.npy",mode='w+',dtype=np.uint16,shape=(n,256,256))
	trc = np.lib.format.open_memmap(f"{cache_dir}/{split}_trc.npy",mode='w+',dtype=np.uint16,shape=(n,256,256))

	for i,c in enumerate(chips):
		rgb[i] = tiff.imread(f"{chip_dir}/{c}_rgb.tif")
		lbl[i] = tiff.imread(f"{chip_dir}/{c}_lbl.tif")
		trc[i] = tiff.imread(f"{feat_dir}/{c}_ftr.tif")[...,3]


	for a in (rgb,lbl,trc):
		a.flush()

	with open(f"{cache_dir}/{split}_ids.txt",'w') as fp:
		fp.write('\n'.join(chips))


################################################################################
# AUGMENTATIONS
################################################################################
class GainScaling:
	def __init__(self,factor=0.05):
		self.hi = 1+factor
		self.lo = 1-factor

	def __call__(self,x):
		return x * torch.empty(x.shape[0],1,1).uniform_(self.lo,self.hi)


class BandOffset: # redundant?
	def __init__(self,factor=0.05):
		self.factor = factor

	def __call__(self,x):
		if random.random() > 0.5:
			return x - self.factor
		return x + self.factor


class GaussianNoise:
	def __init__(self,sigma=0.03,p=0.5):
		self.sigma = sigma
		self.p     = p

	def __call__(self,x):
		if random.random() > self.p:
			return x
		return x + torch.randn_like(x)*self.sigma


class PatchViewTransform:
	'''
	Does not rotate to keep patches in correct order for C matrix calculation.
	This means matching corresponding elements along N in [B,N,D]. N=HxW
	'''
	def __init__(self,blur=0.1):
		self.spectral = v2.Compose([
			v2.RandomApply([v2.GaussianBlur(kernel_size=9,sigma=(0.1,1.0))],p=blur),
			GainScaling(factor=0.05),
		])

		self.post_norm = v2.Compose([
			# BandOffset(factor=0.05),
			GaussianNoise(sigma=0.03,p=0.5)
		])

	def __call__(self,x):
		x = self.spectral(x/255)
		x = (x*255 - MEAN) / STDD
		x = self.post_norm(x)
		return x


class RotatedPatchViewTransform:
	'''
	Passes both img and index label to reconstruct rotation of elements in N.
	Returns (x, patch_ids): patch_ids[i,j] is the original patch index (0..G*G-1)
	that feature-map cell (i,j) of the encoder output came from.
	'''
	def __init__(self,blur=0.1,img_size=256,patch_size=32):
		assert img_size % patch_size == 0, "img_size must be a multiple of patch_size"

		self.P = patch_size           # = encoder stride (5 stride-2 convs -> 32)
		self.G = img_size//patch_size # patches per side on the full chip

		# PIXEL-LEVEL MAP OF PATCH INDICES [H,W]: every pixel holds the id of its PxP patch
		idx = torch.arange(self.G*self.G).view(self.G,self.G)
		idx = idx.repeat_interleave(self.P,0).repeat_interleave(self.P,1)
		self.idx = tv_tensors.Mask(idx)

		# NO CROP: full chip, so every patch appears in both views
		self.geometric = v2.Compose([
			v2.RandomHorizontalFlip(p=0.5),
			v2.RandomVerticalFlip(p=0.5),
			v2.RandomChoice([
				v2.RandomRotation([0,0]),
				v2.RandomRotation([90,90]),
				v2.RandomRotation([180,180]),
				v2.RandomRotation([270,270])
			])			
		])

		self.spectral = v2.Compose([
			v2.RandomApply([v2.GaussianBlur(kernel_size=9,sigma=(0.1,1.0))],p=blur),
			GainScaling(factor=0.05),
		])

		self.post_norm = v2.Compose([
			# BandOffset(factor=0.05),
			GaussianNoise(sigma=0.03,p=0.5)
		])

	def __call__(self,x):

		# # PATCH-ALIGNED RANDOM CROP, SAME WINDOW ON IMAGE AND INDEX MASK
		# n    = self.crop_size//self.P
		# top  = random.randint(0,self.G-n)*self.P
		# left = random.randint(0,self.G-n)*self.P
		# x    = v2.functional.crop(x,top,left,self.crop_size,self.crop_size)
		# mask = v2.functional.crop(self.idx,top,left,self.crop_size,self.crop_size)

		# SAME FLIPS/ROTATION APPLIED TO IMAGE AND INDEX MASK (Mask uses nearest -> ids stay exact)
		x,mask = self.geometric(x,self.idx)

		# ONE ID PER PxP CELL (sample cell centers): [H,W] -> [G,G]
		patch_ids = mask[self.P//2::self.P,self.P//2::self.P].as_subclass(torch.Tensor)

		x = self.spectral(x/255)
		x = (x*255 - MEAN) / STDD
		x = self.post_norm(x)
		return x,patch_ids


class ImageViewTransform:
	'''
	Image-level embeddings.
	'''
	def __init__(self,crop_size=224,blur=0.1):
		self.geometric = v2.Compose([
			v2.RandomCrop(size=crop_size),			
			v2.RandomHorizontalFlip(p=0.5),
			v2.RandomVerticalFlip(p=0.5),
			v2.RandomChoice([
				v2.RandomRotation([0,0]),
				v2.RandomRotation([90,90]),
				v2.RandomRotation([180,180]),
				v2.RandomRotation([270,270])
			])			
		])

		self.spectral = v2.Compose([
			v2.RandomApply([v2.GaussianBlur(kernel_size=7,sigma=(0.5,1.0))],p=blur),
			GainScaling(factor=0.10),
		])

		# self.post_norm = v2.Compose([
			# BandOffset(factor=0.05),
			# GaussianNoise(sigma=0.03,p=0.5)
		# ])

	def __call__(self,x):
		x = self.geometric(x)
		x = self.spectral(x/255) #applied to range 0-1
		x = (x*255 - MEAN)/STDD #[0-255] to ~[-3,3]
		# x = self.post_norm(x) #applied to normalized values 
		return x


class PatchTransform:
	def __init__(self):
		self.view_1 = PatchViewTransform(blur=0.1)
		self.view_2 = PatchViewTransform(blur=0.1)

	def __call__(self,x):
		return self.view_1(x),self.view_2(x)


class RotatedPatchTransform:
	def __init__(self):
		self.view_1 = RotatedPatchViewTransform(blur=0.1)
		self.view_2 = RotatedPatchViewTransform(blur=0.1)

	def __call__(self,x):
		return self.view_1(x),self.view_2(x)


class ImageTransform:
	def __init__(self):
		self.view_1 = ImageViewTransform(crop_size=192,blur=0.1)
		self.view_2 = ImageViewTransform(crop_size=192,blur=1.0) #asymmetric blur

	def __call__(self,x):
		return self.view_1(x),self.view_2(x)


class GPUImageViewTransform(torch.nn.Module):

	'''
	On GPU transform.
	Use:

	aug = dataloader.FastImageViewTransform().to(device)
	for x in dataloader:
		x = x.to(device)
		X1,X2 = aug(x),aug(x)
	'''
	def __init__(self):
		super().__init__()
		self.crop = 192
		self.blur = 0.1
		self.ksize = 9
		self.sigma = (0.1,1.0)

		self.gain = 0.05
		self.offset = 0.05
		self.noise  = 0.03
		self.noise_p = 0.5

		self.register_buffer('mean',MEAN.clone())
		self.register_buffer('stdd',STDD.clone())

	def geometric(self,x):
		# CROP + FLIP
		B,C,H,W = x.shape
		S,dev = self.crop,x.device
		top = torch.randint(0,H-S+1,(B,1),device=dev)
		left = torch.randint(0,W-S+1,(B,1),device=dev)
		i = torch.arange(S,device=dev).expand(B,S)

		#FLIP
		flip_v,flip_h,transp = (torch.rand(3,B,1,device=dev) < 0.5)
		rows = top  + torch.where(flip_v,S-1-i,i)
		cols = left + torch.where(flip_h,S-1-i,i)
		R  = rows[:,:,None].expand(B,S,S)
		cc = cols[:,None,:].expand(B,S,S)
		t  = transp[:,:,None]
		R,cc = torch.where(t,R.transpose(1,2),R),torch.where(t,cc.transpose(1,2),cc)
		b  = torch.arange(B,device=dev)[:,None,None]
		return x[b,:,R,cc].permute(0,3,1,2)


	def gaussian_blur(self,x):
		# PER-SAMPLE SIGMA, ONLY ON THE SAMPLES PICKED
		B,C,H,W = x.shape
		pick = torch.rand(B,device=x.device) < self.blur
		n = int(pick.sum())
		if n == 0:
		        return x
		sig = torch.empty(n,1,device=x.device).uniform_(*self.sigma)
		k   = torch.arange(self.ksize,device=x.device) - (self.ksize-1)/2
		w   = torch.exp(-0.5*(k/sig)**2); w = w/w.sum(1,keepdim=True)
		w   = w.repeat_interleave(C,0)
		p   = self.ksize//2
		y   = F.pad(x[pick].reshape(1,n*C,H,W),(p,p,p,p),mode='reflect')
		y   = F.conv2d(y,w[:,None,None,:],groups=n*C)
		y   = F.conv2d(y,w[:,None,:,None],groups=n*C)
		x   = x.clone(); x[pick] = y.reshape(n,C,H,W)
		return x


	@torch.no_grad()
	def forward(self,x):
		B,dev = x.shape[0],x.device
		x = self.geometric(x).float()
		x = self.gaussian_blur(x)
		
		# GAIN
		x = x * torch.empty(B,1,1,1,device=dev).uniform_(1-self.gain,1+self.gain)

		# NORMALIZE
		x = (x-self.mean)/self.stdd

		# BAND OFFSET
		x = x + self.offset*(torch.randint(0,2,(B,1,1,1),device=dev)*2-1)

		# GAUSSIAN NOISE
		x = x + self.noise * torch.randn_like(x)*(torch.rand(B,1,1,1,device=dev) < self.noise_p)

		return x	

################################################################################
# DATASET CLASSES
################################################################################
class PairedDataset(torch.utils.data.Dataset):
	'''
	Self-supervised pre-training dataloder.
	'''
	def __init__(self,chip_dir,transform,split='train'):
		split_csv = "./split_chips.csv"
		chips,_  = read_split(split_csv,split)
		self.ids = [f"{chip_dir}/{c}" for c in chips]
		self.transform = transform

	def get_bands(self,idx):
		img = torch.from_numpy(tiff.imread(f"{self.ids[idx]}_rgb.tif")).to(torch.float32)
		return tv_tensors.Image(img)

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		return self.transform(self.get_bands(idx))


class LabeledDataset(torch.utils.data.Dataset):
	'''
	Evaluation. Loads, image,label, mask.
	Image: [10,256,256] uint8
	Label: [256,256] uint16 with range 1-999
	Mask: [4,256,256] uint16 with bands [RUCA,population,area,tract_ids]
	'''
	def __init__(self,chip_dir,feat_dir,split='val'):
		# DIR PATHS
		chips,self.n_tracts = read_split('./split_chips.csv',split)
		self.band_ids = [f"{chip_dir}/{c}" for c in chips]
		self.feat_ids = [f"{feat_dir}/{c}" for c in chips]

	def get_bands(self,idx):
		# return Image.open(f'{self.ids[idx]}_rgb.tif')
		img = torch.from_numpy(tiff.imread(f"{self.band_ids[idx]}_rgb.tif")).to(torch.float32)
		img = tv_tensors.Image(img)
		return img

	def get_label(self,idx):
		# single decimal percentage (00.0%) stored as 0-999, return as [0-1]
		lbl = torch.from_numpy(tiff.imread(f"{self.band_ids[idx]}_lbl.tif"))
		lbl = torch.squeeze(lbl,0).to(torch.float32).div(1000) 
		return lbl

	def get_tract(self,idx):
		ext = torch.from_numpy(tiff.imread(f"{self.feat_ids[idx]}_ftr.tif")[...,3])
		ext = torch.squeeze(ext,0).to(torch.uint16)
		return ext

	def __len__(self):
		return len(self.band_ids)

	def __getitem__(self,idx):

		# LOAD TO TENSORS
		bands = self.get_bands(idx)
		label = self.get_label(idx)
		extra = self.get_tract(idx)

		# NORMALIZE BANDS
		bands = (bands - MEAN) / STDD

		# DONE
		return bands,label,extra


class MappedPairedDataset(torch.utils.data.Dataset):
	'''
	Memory mapped to .npy for fast loading. Loads from disk.
	'''
	def __init__(self,cache_dir,transform,split='train'):
		self.path = f"{cache_dir}/{split}_rgb.npy"
		self.n    = np.load(self.path,mmap_mode='r').shape[0]
		self.rgb  = None
		self.transform = transform

	def __len__(self):
		return self.n

	def __getitem__(self,idx):
		if self.rgb is None:
			self.rgb = np.load(self.path,mmap_mode='r')
		img = torch.from_numpy(np.array(self.rgb[idx])).to(torch.float32)
		return self.transform(tv_tensors.Image(img))


class MappedLabeledDataset(torch.utils.data.Dataset):
	'''
	Memory mapped to .npy in disk for fast loading. Loads from disk.
	Reduces cpu-gpu transfer size by loading and passing original files and 
	doing augmentations only in GPU.
	'''
	def __init__(self,cache_dir,split='val'):
		self.chips,self.n_tracts = read_split('./split_chips.csv',split)		
		self.dir   = cache_dir
		self.split = split
		self.n     = np.load(f"{self.dir}/{self.split}_lbl.npy",mmap_mode='r').shape[0]

		self.rgb = None
		self.lbl = None
		self.trc = None

	def __len__(self):
		return self.n

	def __getitem__(self,idx):
		if self.rgb is None:
			self.rgb = np.load(f"{self.dir}/{self.split}_rgb.npy",mmap_mode='r')
			self.lbl = np.load(f"{self.dir}/{self.split}_lbl.npy",mmap_mode='r')
			self.trc = np.load(f"{self.dir}/{self.split}_trc.npy",mmap_mode='r')
		bands = torch.from_numpy(np.array(self.rgb[idx])).to(torch.float32)
		label = torch.from_numpy(np.array(self.lbl[idx])).to(torch.float32).div(1000)
		tract = torch.from_numpy(np.array(self.trc[idx]))
		return (bands-MEAN)/ STDD, label, tract


class RAMPairedDataset(torch.utils.data.Dataset):
	def __init__(self,cache_dir,transform,split='train'):
		self.path = f"{cache_dir}/{split}_rgb.npy"
		self.transform = transform

		print("Loading cached dataset to RAM...")
		self.rgb = torch.from_numpy(np.load(self.path))
		self.n   = self.rgb.shape[0]
		print("Loaded to RAM")


	def __len__(self):
		return self.n


	def __getitem__(self,idx):
		img = self.rgb[idx].to(torch.float32)
		return self.transform(tv_tensors.Image(img))


class RAMLabeledDataset(torch.utils.data.Dataset):

	def __init__(self,cache_dir,split='val'):

		self.chips,self.n_tracts = read_split('./split_chips.csv',split)

		# LOAD DATASET
		print("Loading cached dataset to RAM...")
		self.rgb = torch.from_numpy(np.load(f"{cache_dir}/{split}_rgb.npy"))
		self.lbl = torch.from_numpy(np.load(f"{cache_dir}/{split}_lbl.npy"))
		self.trc = torch.from_numpy(np.load(f"{cache_dir}/{split}_trc.npy"))
		print("Loaded to RAM")

		# SIZE
		self.n = self.rgb.shape[0]


	def __len__(self):
		return self.n


	def __getitem__(self,idx):

		# LOAD FROM ARRAY IN RAM
		bands = self.rgb[idx]
		label = self.lbl[idx]
		tract = self.trc[idx]

		# ADJUST
		bands = tv_tensors.Image(bands.to(torch.float32))
		label = torch.squeeze(label,0).to(torch.float32).div(1000)
		tract = torch.squeeze(tract,0).to(torch.uint16)

		# NORMALIZE BANDS
		bands = (bands-MEAN)/STDD

		return bands, label, tract

################################################################################
# OTHER
################################################################################


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Barlow Twins Dataloaders for Sentinel-2 chips.')
	parser.add_argument('--data-dir',type=str,required=True,help='Dir for chips.')
	parser.add_argument('--mask-dir',type=str,required=True,help='Dir for feature/tract masks (*_ftr.tif).')
	parser.add_argument('--cache-dir',type=str,required=True,help='Output dir')
	parser.add_argument('--split',type=str,required=True,help='train/val/test')

	args = parser.parse_args()

	assert os.path.isdir(args.data_dir), f"Data dir not found at {args.data_dir}"
	assert os.path.isdir(args.mask_dir), f"Data dir not found at {args.mask_dir}"
	assert os.path.isdir(args.cache_dir), f"Data dir not found at {args.cache_dir}"
	args.data_dir = args.data_dir.rstrip('/')
	args.mask_dir = args.mask_dir.rstrip('/')
	args.cache_dir = args.cache_dir.rstrip('/')

	build_cached_dataset(args.data_dir,args.mask_dir,'./split_chips.csv',args.split,args.cache_dir)
