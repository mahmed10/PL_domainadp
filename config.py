import argparse
import os
import torch
import data
from models.unet import UNET

parser = argparse.ArgumentParser(description='Benchmarking')
parser.add_argument('--da_mode', type=str, default='SingleDA_')
# parser.add_argument('--da_mode', type=str, default='Sourceonly_')
parser.add_argument('--branch_mode', type=str, default='SingleBranch_')
parser.add_argument('--class_mode', type=str, default='CommonClasses_')
parser.add_argument('--source_dataset', type=str, default=['cityscapes'])
parser.add_argument('--target_dataset', type=str, default=['cityscapes'])
parser.add_argument('--other_mode', type=str, default = '')
parser.add_argument('--train_mode', type=str, default='test')
parser.add_argument('--train_path_list', type=str, default=['./dataset/CityScapes/trainlist_hardrmi20percent.txt'])
parser.add_argument('--val_path_list', type=str, default=['./dataset/CityScapes/trainlist1.txt'])
parser.add_argument('--data_mode', type=str, default='rgb')
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--image_width', type=int, default=572)
parser.add_argument('--image_height', type=int, default=572)
parser.add_argument('--num_classes', type=int, default=20)
parser.add_argument('--train_data_size', type=int, default=0)
parser.add_argument('--model_path', type=str, default='./checkpoints/')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--multi_level', type=bool, default=False)

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--start_epoches', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.0005)

args = parser.parse_args()

model_path = args.model_path+args.da_mode+args.branch_mode+args.class_mode+'_Source_'+args.source_dataset[0]+'_Target_'+args.target_dataset[0]+args.other_mode

try:
	os.mkdir(model_path)
except OSError:
	print(os.path.abspath(model_path) + ' folder already existed')
else:
	print(os.path.abspath(model_path) + ' folder created')
args.model_path = model_path+'/'
del model_path


if torch.cuda.is_available():
	args.device = 'cuda:0'
	torch.cuda.empty_cache()
else:
	args.device = "cpu"

args.model = UNET(in_channels=args.in_channels, classes=args.num_classes).to(args.device)
args.source_loader = data.setup_loaders(args.source_dataset, args.train_path_list, args.batch_size)
args.target_loader = data.setup_loaders(args.target_dataset, args.val_path_list, args.batch_size)