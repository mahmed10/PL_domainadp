import os
import numpy as np
import torch
from PIL import Image
import yaml
import data.edge_utils as edge_utils


class Relis3D(torch.utils.data.Dataset):
	def __init__(self, train, path_list, data_mode):
		self.train = train
		self.data_mode = data_mode

		with open(path_list, "r") as file:
			self.imgs = file.readlines()
		#print("\nImage path files:")
		#print(self.imgs)

		self.masks = [
			path.replace("pylon_camera_node", "pylon_camera_node_label_id")
			.replace(".jpg", ".png")
			for path in self.imgs
		]
		#print("\nImage mask path files:")
		#print(self.masks)

		self.binfiles = [
			path.replace("Rellis_3D_image_example", "Rellis_3D_lidar_example")
			.replace("pylon_camera_node", "os1_cloud_node_kitti_bin")
			.replace(".jpg", ".bin")
			for path in self.imgs
		]
		#print("\nLidar path files:")
		#print(self.binfiles)

		self.labelfiles = [
			path.replace("Rellis_3D_image_example", "Rellis_3D_lidar_example")
			.replace("pylon_camera_node", "os1_cloud_node_semantickitti_label_id")
			.replace(".jpg", ".label")
			for path in self.imgs
		]
		#print("\nLidar mask path files:")
		#print(self.labelfiles)

		with open("./dataset/Rellis_3D.yaml", 'r') as stream:
			relis3dyaml = yaml.safe_load(stream)
		self.learning_map = relis3dyaml['learning_map']
		#print("\n")
        

	def __getitem__(self, index):
		if (self.data_mode != 'lidar'):
			img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip() 
			#print("img_path path: ", img_path)
			#print("mask_path path: ", mask_path)
			img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
			if(self.data_mode == 'rgb'):
				return img, mask

		if (self.data_mode != 'rgb'):
			binfile_path, labelfile_path = self.binfiles[index].rstrip(), self.labelfiles[index].rstrip() 
			#print("binfile_path path: ", binfile_path)
			#print("labelfile_path path: ", labelfile_path)
			binfile = np.fromfile(binfile_path, dtype=np.float32).reshape((-1, 4))
            
			labelfile = np.fromfile(labelfile_path, dtype=np.int32).reshape((-1,1))
			labelfile = labelfile & 0xFFFF #delete high 16 digits binary
			labelfile = np.vectorize(self.learning_map.__getitem__)(labelfile)
			labelfile = labelfile.astype(np.uint8)

			if(self.data_mode == 'lidar'):
				return binfile, labelfile

		return img, binfile, mask, labelfile

	def __len__(self):
		return len(self.imgs)
