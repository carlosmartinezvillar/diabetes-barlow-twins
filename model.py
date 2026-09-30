import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

############################################################
# CONV BLOCKS
############################################################
class ConvBlock(nn.Module):
	def __init__(self,channels,depth=2):
		super().__init__()
		self.block = nn.ModuleList()
		for i in range(depth):
			self.block.append(nn.Sequential(
				nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
				nn.GroupNorm(1,channels),
				nn.GELU()
			))

	def forward(self,x):
		out = x
		for layer in self.block:
			out = layer(out)
		return x + out

############################################################
# ATTENTION BLOCKS
############################################################
class MultiHeadSelfAttention(nn.Module):
	'''
	Multi-head Self-Attention
	B: batch dimension
	D: embedding dimension
	N: sequence length
	H: head dimension
	'''
	def __init__(self, D, num_heads=4):
		super().__init__()

		#dimensions
		assert D % num_heads == 0, f"channels={E} not divisible by num_heads={num_heads}"
		self.D         = D
		self.num_heads = num_heads
		self.H         = D // num_heads

		# Weight matrices
		self.W_qkv  = nn.Linear(D, D * 3, bias=False)
		self.W_o    = nn.Linear(D, D, bias=False)


	def forward(self, x):
		B,N,_ = x.shape
		QKV = self.W_qkv(x)         # [B,N,3D]
		Q,K,V = QKV.chunk(3,dim=-1) # each [B,N,D]
		Q = Q.view(B,N,self.num_heads,self.H).transpose(1,2) # [B,num_heads,N,H]
		K = K.view(B,N,self.num_heads,self.H).transpose(1,2)
		V = V.view(B,N,self.num_heads,self.H).transpose(1,2)

		# SAME AS softmax(QK^T/sqrt(H))V
		x = F.scaled_dot_product_attention(Q,K,V) # [B,num_heads,N,H]
		x = x.transpose(1, 2).reshape(B,N,self.D) #[B,num_heads,N,H] -> [B,N,num_heads,H] -> [B,N,D]
		return self.W_o(x) # [B,N,D]


class MLP(nn.Module):
	'''
	Vanilla MLP layer in transformer block.
	Just consistency when printing really...
	'''
	def __init__(self, dim, mlp_ratio=4):
		super().__init__()
		hidden_dim = dim * mlp_ratio
		self.layers = nn.Sequential(
		    nn.Linear(dim, hidden_dim),
		    nn.GELU(),
		    nn.Linear(hidden_dim, dim)
		)


	def forward(self, x):
		return self.layers(x)


class ViTLayer(nn.Module):
	'''
	A ViT layer (i.e. Multi-head Self-attention + MLP)
	'''
	def __init__(self,D,num_heads,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(D)
		self.attn  = MultiHeadSelfAttention(D,num_heads)
		self.norm2 = nn.LayerNorm(D)
		self.mlp   = MLP(D, mlp_ratio)

	def forward(self, tokens):
		tokens = tokens + self.attn(self.norm1(tokens))
		tokens = tokens + self.mlp(self.norm2(tokens))
		return tokens


class ViTBlock(nn.Module):
	'''
	Wrapper for self-attention s.t.: [B,C,H,W] -> [B,N,D] -> [B,C,H,W]
	Used when ViT block is in 'intermediate' layers instead of only in last layer.
	'''	
	def __init__(self,E,num_heads,mlp_ratio=4,depth=1):
		super().__init__()
		self.block = nn.ModuleList([ViTLayer(E,num_heads,mlp_ratio) for _ in range(depth)])

	def forward(self,x):
		B,C,H,W = x.shape
		tokens = x.permute(0,2,3,1).reshape(B,H*W,C)
		for layer in self.block:
			tokens = layer(tokens)
		return tokens.reshape(B,H,W,C).permute(0,3,1,2)


############################################################
# ENCODERS
############################################################
class HybridEncoder(nn.Module):
	'''
	Default encoder.
	'''
	def __init__(self,init_channels=64,cnn_depth=2):
		super().__init__()

		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		self.conv1 = ConvBlock(channels=init_channels,depth=cnn_depth)
		self.down1 = nn.Conv2d(init_channels,init_channels*2,**down_params)

		self.conv2 = ConvBlock(channels=init_channels*2,depth=cnn_depth)
		self.down2 = nn.Conv2d(init_channels*2,init_channels*4,**down_params)

		self.conv3 = ConvBlock(channels=init_channels*4,depth=cnn_depth)
		self.down3 = nn.Conv2d(init_channels*4,init_channels*8,**down_params)

		self.conv4 = ConvBlock(channels=init_channels*8,depth=cnn_depth)
		self.down4 = nn.Conv2d(init_channels*8,init_channels*16,**down_params)

		self.attn1 = ViTLayer(init_channels*16,num_heads=16,mlp_ratio=4)
		self.attn2 = ViTLayer(init_channels*16,num_heads=16,mlp_ratio=4)


	def forward(self,x):

		# CONV LAYERS
		x = self.down1(self.conv1(x))
		x = self.down2(self.conv2(x))
		x = self.down3(self.conv3(x))
		x = self.down4(self.conv4(x))

		# SINGLE (DOUBLE?) ATTENTION LAYER
		B,C,H,W = x.shape
		x = x.permute(0,2,3,1).reshape(B,H*W,C)
		x = self.attn1(x)
		x = self.attn2(x)

		# RETURN [B,N,D]
		return x


class PureConvEncoder(nn.Module):

	def __init__(self):
		super().__init__()
		pass


	def forward(self,x):
		# CONV LAYERS

		# [B,C,H,W] -> [B,H*W,C]
		return x


class PureAttnEncoder(nn.Module):

	def __init__(self):
		super().__init__()
		pass


	def forward(self,x):

		# PATCH EMBED

		# ATTENTION LAYERS

		return x

############################################################
# BARLOW TWINS
############################################################
class Projector(nn.Module):
	'''
	Barlow Twins projector.
	'''
	def __init__(self,in_dim,out_dim=2048):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(in_dim, out_dim),
			nn.BatchNorm1d(out_dim),
			nn.ReLU(),
			nn.Linear(out_dim, out_dim) # Final projection dimension
		)
	def forward(self,x):
		return self.layers(x)


class BarlowTwinsLoss(nn.Module):
	'''
	L = sum_i (1 - C_ii)^2 + lambda * sum_i sum_{j!=i} C_ij^2
	C is the cross-correlation matrix between the batch-standardized embeddings
	of the two views. Computed in float32 regardless of autocast.
	'''
	def __init__(self,proj_dim,lambd=5e-3):
		super().__init__()
		self.lambd = lambd
		self.bn    = nn.BatchNorm1d(proj_dim,affine=False,track_running_stats=False)

	def forward(self,z1,z2):
		with torch.autocast(device_type='cuda' if z1.is_cuda else 'cpu',enabled=False):
			z1 = self.bn(z1.float())
			z2 = self.bn(z2.float())
			c  = z1.T @ z2 / z1.shape[0] # [D,D]

			diag     = torch.diagonal(c)
			on_diag  = (diag - 1).pow(2).sum()
			off_diag = c.pow(2).sum() - diag.pow(2).sum()
			loss     = on_diag + self.lambd * off_diag
		return loss, on_diag.detach(), off_diag.detach()



class SentinelBarlowTwins(nn.Module):
	'''
	Main class/entry point.
	'''
	def __init__(self,model_id,in_chan=12,block_chan=64,conv_per_block=2):
		super().__init__()

		# META
		self.model_name = "S2BT"
		self.model_id   = model_id

		# Input channels projection [B,S2 12-band,H,W] -> [B,block_chan,H,W] 
		self.in_layer = nn.Conv2d(in_chan,block_chan,3,1,1,bias=True)

		# ENCODERS
		self.E1 = HybridEncoder(init_channels=block_chan,cnn_depth=conv_per_block)
		self.E2 = HybridEncoder(init_channels=block_chan,cnn_depth=conv_per_block)

		# PROJECTOR
		proj_dim = 2048
		self.projector1 = Projector(block_chan*16,proj_dim)
		self.projector2 = Projector(block_chan*16,proj_dim)

		# LOSS
		lambd = 5e-3
		self.loss = BarlowTwinsLoss(proj_dim,lambd)


	def forward(self,x,x_aug):

		normal    = self.E1(x)
		augmented = self.E2(x_aug)

		B,N,D = normal.shape
		normal = normal.reshape(B*N,D)
		augmented = augmented.reshape(B*N,D)

		normal    = self.projector1(normal)
		augmented = self.projector2(augmented)

		return normal,augmented


############################################################
# PREDICTION
############################################################
class MaskDownsampler(nn.Module):
	'''
	Calculate average over a 31x31 area of pixels in masks. 
	This converts a 256x256 mask into a 16x16 tensor, keeping correct relative 
	pixel positions for direct comparison to output embeddings.
	
	'''
	def __init__(self):
		# [B,1,256,256] -> [B,1,16,16]
		self.downsampler = nn.AvgPool2d(kernel_size=31,stride=16,padding=15)

	def forward(self,x)
		return self.downsample(x) #i.e. targets


class HealthVariablePredictorHead(nn.Module):

	def __init__(self,embedding_dim=512):
		super().__init__()
		# A 1x1 convolution/linear regression head over each 16x16 cell
		self.predict_head = nn.Sequential(
			nn.Conv2d(embedding_dim, 128, kernel_size=1),
			nn.ReLU(),
			nn.Conv2d(128, 1, kernel_size=1)
		)

	def forward(self,x):
		return self.predict_head(x)


class MaskedSpatialDownsampler:
	'''
	Might need masking if data not adjusted.
	'''
	def __init__(self, kernel_size=31, stride=16, padding=15):
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		# Create a fixed uniform weight kernel for local summing
		# Shape: (1, 1, K, K) filled with 1.0
		self.weight = torch.ones((1, 1, kernel_size, kernel_size))

	def __call__(self, health_map_256):
		# Move the weight kernel to the same device as your input data
		weight = self.weight.to(health_map_256.device)

		# 1. Create a binary mask where valid data coverage exists
		# 1.0 means valid data, 0.0 means missing coverage area
		mask = (health_map_256 != 0.0).float()

		# 2. Local sum of actual health values within the receptive field
		# We zero out any background noise just in case by multiplying by mask
		masked_health = health_map_256 * mask
		sum_health = F.conv2d(masked_health, weight, stride=self.stride, padding=self.padding)

		# 3. Local count of valid pixels within the receptive field
		count_valid = F.conv2d(mask, weight, stride=self.stride, padding=self.padding)

		# 4. Compute the true coverage average, avoiding division by zero
		# If a neighborhood has 0 valid pixels, keep the result as 0.0
		targets_downsampled = torch.where(
			count_valid > 0, 
			sum_health / (count_valid + 1e-8), 
			torch.zeros_like(sum_health)
		)

	return targets_downsampled



if __name__ == '__main__':
	pass
	#do some checks/print