import torch
import torchvision
from torchvision import tv_tensors
from PIL import Image
import torchvision.transforms as v1
import torchvision.transforms.v2 as v2
import glob
import os

# DATASET MEAN AND STD DEV.
MEAN = torch.tensor([0,0,0]).view(-1,1,1)
STDD = torch.tensor([0,0,0]).view(-1,1,1)

################################################################################
# AUGMENTATIONS
################################################################################


################################################################################
# DATASET CLASSES
################################################################################
class SentinelDataset(torch.utils.data.Dataset):

	def __init__(self,chip_dir,transform=None):

		self.dir        = chip_dir
		self.band_files = sorted(glob.glob(f"{chip_dir}/*_bands.tif"))
		self.ids        = [i[0:-8] for i in self.band_files]

		self.train_transform = transform


	def get_bands(self,idx):
		return Image.open(f'{self.ids[idx]}_bands.tif')


	def get_label(self,idx):
		lbl = v2.functional.pil_to_tensor(Image.open(f'{self.ids[idx]}_label.tif'))
		lbl = torch.squeeze(lbl,0).to(torch.float32)
		return lbl


	def __len__(self):
		return len(self.ids)


	def __getitem__(self,idx):

		bands = self.load_bands(idx)
		label = self.load_label(idx)

		if self.train_transform:
			bands,label = self.train_transform(bands,label)

		bands = (bands - MEAN) / STDD

		return bands,bands_augmented,label


class FastSentinelDataset():
	'''
	Memory mapped to .npy for fast loading.
	'''
	def __init__(self):
		pass

	def __len__(self):
		return 0

	def __getitem__(self,idx):
		return idx,idx


################################################################################
# OTHER
################################################################################
def create_cached_dataset():
	pass