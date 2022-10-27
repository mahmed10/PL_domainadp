import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
#from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from domain_adaptation.discriminator import get_fc_discriminator
from domain_adaptation.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from domain_adaptation.utils.func import loss_calc, bce_loss
from domain_adaptation.utils.loss import entropy_loss
from domain_adaptation.utils.func import prob_2_entropy
from domain_adaptation.utils.viz_segmask import colorize_mask

from evaluate import iou_calculation

def train_advent(args):
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
	strating_epoch = 0
	train_data_size = len(trainloader) if args.train_data_size == 0 else args.train_data_size

	# Create the model and start the training.
	# SEGMNETATION NETWORK
	model.train()
	model.to(device)
	cudnn.benchmark = True
	cudnn.enabled = True

	# DISCRIMINATOR NETWORK
	# feature-level
	d_aux = get_fc_discriminator(num_classes=num_classes)
	d_aux.train()
	d_aux.to(device)

	# seg maps, i.e. output, level
	d_main = get_fc_discriminator(num_classes=num_classes)
	d_main.train()
	d_main.to(device)

	# OPTIMIZERS
	# segnet's optimizer
	optimizer = optim.SGD(model.parameters(),
						lr=0.001, #2.5e-4,
						momentum=0.9,
						weight_decay=0.00005)

	# discriminators' optimizers
	optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=1e-4, betas=(0.9, 0.99))
	optimizer_d_main = optim.Adam(d_main.parameters(), lr=1e-4, betas=(0.9, 0.99))

	#load model
	train_ious = []
	val_ious = []
	if args.load_model == True:
		checkpoint = torch.load(model_path+'e_'+str(args.start_epoches-1).zfill(4))
		model.load_state_dict(checkpoint['model_state_dict'])
		d_main.load_state_dict(checkpoint['d_main_state_dict'])
		d_aux.load_state_dict(checkpoint['d_aux_state_dict'])
		optimizer.load_state_dict(checkpoint['optim_state_dict'])
		optimizer_d_aux.load_state_dict(checkpoint['optim_d_aux_dict'])
		optimizer_d_main.load_state_dict(checkpoint['optim_d_main_dict'])
		strating_epoch = checkpoint['epoch']+1
		train_ious = checkpoint['train_iou']
		val_ious = checkpoint['val_iou']
		print("Model successfully loaded!")

	# interpolate output segmaps
	interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
	interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

	# labels for adversarial training
	source_label = 0
	target_label = 1
	# import pdb; pdb.set_trace()
	for e_epoch in tqdm(range(args.epoches)):
		if (e_epoch <strating_epoch):
			continue
		trainloader_iter = enumerate(trainloader)
		targetloader_iter = enumerate(targetloader)
		train_iou = 0
		val_iou = 0
		for i_iter in tqdm(range(train_data_size)):
		# reset optimizers
			optimizer.zero_grad()
			optimizer_d_aux.zero_grad()
			optimizer_d_main.zero_grad()
			# adapt LR if needed
			adjust_learning_rate(optimizer, i_iter)
			adjust_learning_rate_discriminator(optimizer_d_aux, i_iter)
			adjust_learning_rate_discriminator(optimizer_d_main, i_iter)

			# UDA Training
			# only train segnet. Don't accumulate grads in disciminators
			for param in d_aux.parameters():
				param.requires_grad = False
			for param in d_main.parameters():
				param.requires_grad = False
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

			# adversarial training ot fool the discriminator
			_, batch = targetloader_iter.__next__()
			if (args.train_mode == 'train'):
				images, _= batch
			elif (args.train_mode == 'test'):
				images= batch
			val_iou += iou_calculation(batch, model, device) if args.train_mode == 'train' else 0
			pred_trg_aux, pred_trg_main = model(images.cuda(device))
			if multi_level:
				pred_trg_aux = interp_target(pred_trg_aux)
				d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
				loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
			else:
				loss_adv_trg_aux = 0
			pred_trg_main = interp_target(pred_trg_main)
			d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
			loss_adv_trg_main = bce_loss(d_out_main, source_label)
			loss = (0.001 * loss_adv_trg_main + 0.0002 * loss_adv_trg_aux)
			loss = loss
			loss.backward()

			# Train discriminator networks
			# enable training mode on discriminator networks
			for param in d_aux.parameters():
				param.requires_grad = True
			for param in d_main.parameters():
				param.requires_grad = True
			# train with source
			if multi_level:
				pred_src_aux = pred_src_aux.detach()
				d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
				loss_d_aux = bce_loss(d_out_aux, source_label)
				loss_d_aux = loss_d_aux / 2
				loss_d_aux.backward()
			pred_src_main = pred_src_main.detach()
			d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
			loss_d_main = bce_loss(d_out_main, source_label)
			loss_d_main = loss_d_main / 2
			loss_d_main.backward()

			# train with target
			if multi_level:
				pred_trg_aux = pred_trg_aux.detach()
				d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
				loss_d_aux = bce_loss(d_out_aux, target_label)
				loss_d_aux = loss_d_aux / 2
				loss_d_aux.backward()
			else:
				loss_d_aux = 0
			pred_trg_main = pred_trg_main.detach()
			d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
			loss_d_main = bce_loss(d_out_main, target_label)
			loss_d_main = loss_d_main / 2
			loss_d_main.backward()

			optimizer.step()
			optimizer_d_aux.step()
			optimizer_d_main.step()

			current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
							'loss_seg_src_main': loss_seg_src_main,
							'loss_adv_trg_aux': loss_adv_trg_aux,
							'loss_adv_trg_main': loss_adv_trg_main,
							'loss_d_aux': loss_d_aux,
							'loss_d_main': loss_d_main}
			#print_losses(current_losses, i_iter)

			sys.stdout.flush()
		train_ious.append(train_iou/train_data_size)
		print('train_ious', train_ious)
		val_ious.append(val_iou/train_data_size)
		print('val_ious', val_ious)

		torch.save({
			'model_state_dict': model.state_dict(),
			'd_main_state_dict': d_main.state_dict(),
			'd_aux_state_dict': d_aux.state_dict(),
			'optim_state_dict': optimizer.state_dict(),
			'optim_d_aux_dict': optimizer_d_aux.state_dict(),
			'optim_d_main_dict': optimizer_d_main.state_dict(),
			'epoch': e_epoch,
			'train_iou': train_ious,
			'val_iou' : val_ious
		}, model_path + 'e_'+str(e_epoch).zfill(4))

def print_losses(current_losses, i_iter):
	list_strings = []
	for loss_name, loss_value in current_losses.items():
		list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
	full_string = ' '.join(list_strings)
	tqdm.write(f'iter = {i_iter} {full_string}')



def to_numpy(tensor):
	if isinstance(tensor, (int, float)):
		return tensor
	else:
		return tensor.data.cpu().numpy()


def train_domain_adaptation(args):
	print(args.model_path)
	train_advent(args)