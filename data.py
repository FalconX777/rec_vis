import zipfile
import os
import torch
import numpy as np

import torchvision.transforms as transforms

# Dataset at https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip

class White_noise():
	"""
	Add a random level gausian with noise to the image
	"""
	def __init__(self, level=0.1):
		self.level = level
	
	def __call__(self, img):
		return img+torch.randn_like(img)*(self.level*np.random.rand())

image_size = 341
crop_size = int(image_size/1.14) # = 299

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

data_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
	transforms.RandomCrop((crop_size,crop_size)),
    transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Lambda(White_noise(level = 0.05))
])

val_transforms = transforms.Compose([
	transforms.Resize((crop_size, crop_size)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
