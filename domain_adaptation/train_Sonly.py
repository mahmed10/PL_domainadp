import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from domain_adaptation.utils.func import adjust_learning_rate
from domain_adaptation.utils.func import loss_calc

from evaluate import iou_calculation

from tqdm import tqdm
def train(args):
	#setting up variables
	model = args.model
	trainloader = args.source_loader
	targetloader = args.target_loader
	model_path = args.model_path
	device = args.device
	num_classes = args.num_classes
	input_size_source = (args.image_width,args.image_height)
	input_size_target = (args.image_width,args.image_height)
	multi_level = args.multi_level
	train_data_size = len(trainloader) if args.train_data_size == 0 else args.train_data_size
	strating_epoch = 0

	# Create the model and start the training.
	# SEGMNETATION NETWORK
	model.train()
	model.to(device)
	cudnn.benchmark = True
	cudnn.enabled = True

	# OPTIMIZERS
	# segnet's optimizer
	optimizer = optim.SGD(model.parameters(),
						lr=2.5e-4,
						momentum=0.9,
						weight_decay=0.00005)
	loss_function = nn.CrossEntropyLoss(ignore_index=255)

	if args.load_model == True:
		checkpoint = torch.load(model_path+'e_'+str(args.start_epoches-1).zfill(4))
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optim_state_dict'])
		strating_epoch = checkpoint['epoch']+1
		train_ious = checkpoint['train_iou']
		val_ious = checkpoint['val_iou']
		print("Model successfully loaded!") 

	# interpolate output segmaps
	interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
	interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
	train_ious = []
	val_ious = []
	for e_epoch in tqdm(range(strating_epoch, args.epoches)):
		trainloader_iter = enumerate(trainloader)
		targetloader_iter = enumerate(targetloader)
		train_iou = 0
		val_iou = 0
		for i_iter in tqdm(range(train_data_size)):
			# reset optimizers
			optimizer.zero_grad()
			# adapt LR if needed
			adjust_learning_rate(optimizer, i_iter)
			# train on source
			_, batch = trainloader_iter.__next__()
			images_source, labels = batch
			train_iou += iou_calculation(batch, model, device)
			pred_src_aux, pred_src_main = model(images_source.cuda(device))
			if multi_level:
				pred_src_aux = interp(pred_src_aux)
				loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
			else:
				loss_seg_src_aux = 0
			pred_src_main = interp(pred_src_main)
			loss_seg_src_main = loss_calc(pred_src_main, labels, device)
			loss = (1.0 * loss_seg_src_main + 0.1 * loss_seg_src_aux)
			loss.backward()

			# test on target
			_, batch = targetloader_iter.__next__()
			images, _= batch
			val_iou += iou_calculation(batch, model, device)

			optimizer.step()
			sys.stdout.flush()

		train_ious.append(train_iou/train_data_size)
		print('train_ious', train_ious)
		val_ious.append(val_iou/train_data_size)
		print('val_ious', val_ious)


		torch.save({
			'model_state_dict': model.state_dict(),
			'optim_state_dict': optimizer.state_dict(),
			'epoch': e_epoch,
			'train_iou': train_ious,
			'val_iou' : val_ious
		}, model_path + 'e_'+str(e_epoch).zfill(4))





def train_source_only(args):
	print(args.model_path)
	train(args)