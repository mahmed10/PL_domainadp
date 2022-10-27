import os
import numpy as np
import torch
from PIL import Image
import yaml

from torchvision import transforms, datasets


class GTA5(torch.utils.data.Dataset):
	def __init__(self, path_list, train = 'train', data_mode ='rgb', scale = 1, target_width = 572, target_height = 572):

		self.mode = train
		self.data_mode = data_mode
		self.scale = scale
		self.target_w = target_width
		self.target_h = target_height

		with open(path_list, "r") as file:
			self.imgs = file.readlines()

		self.masks = [
			path.replace("_images/", "_labels/")
			.replace("/images/", "/labels/")
			for path in self.imgs
		]


		with open("./dataset/GTA5/gta5_copy.yaml", 'r') as stream:
			relis3dyaml = yaml.safe_load(stream)
		self.learning_map = relis3dyaml['learning_map']

	def convert_label(self, label, inverse=False):
		temp = label.copy()*255 
		for k, v in self.learning_map.items():
			label[temp== k] = v
		return label

	def __getitem__(self, index):
		return self.supdata(index = index)

	def supdata(self, index):
		img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()

		img = Image.open(img_path).convert('RGB')
		mask = Image.open(mask_path)

		w,h = img.size
		target_w, target_h = int(w * self.scale), int(h * self.scale)
		target_w, target_h = self.target_w, self.target_h
		img = img.resize((target_w, target_h))
		img = transforms.ToTensor()(img)

		mask = mask.resize((target_w, target_h))
		mask = transforms.ToTensor()(mask)

		mask = np.array(mask)
		mask = mask [0,:,:]

		mask = self.convert_label(mask)
		mask = mask.astype(np.uint8)

		img = np.asarray(img) 
		
		return torch.as_tensor(img.copy()).float().contiguous(),torch.as_tensor(mask.copy()).long().contiguous()

	def unsupdata(self, index):
		img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()

		img = Image.open(img_path).convert('RGB')
		mask = Image.open(mask_path)

		w,h = img.size
		target_w, target_h = int(w * self.scale), int(h * self.scale)
		target_w, target_h = self.target_w, self.target_h
		img = img.resize((target_w, target_h))
		img = transforms.ToTensor()(img)
		# img1 = img.÷rotate(90)

		mask = mask.resize((target_w, target_h))
		mask = transforms.ToTensor()(mask)

		mask = np.array(mask)
		mask = mask [0,:,:]

		mask = self.convert_label(mask)
		mask = mask.astype(np.uint8)

		img = np.asarray(img)
		# img1 = np.rot90(img,axes=(-2,-1))
		
		return torch.as_tensor(img.copy()).float().contiguous(),torch.as_tensor(mask.copy()).long().contiguous()

	def __len__(self):
		return len(self.imgs)