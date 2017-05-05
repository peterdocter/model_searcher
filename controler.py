import argparse
parser = argparse.ArgumentParser(description='train a speech classifer')
parser.add_argument('--log-dir', type=str, 
		default='hdfs://yz-cpu-vm001.hogpu.cc:8020/user/great_searcher/log/',
		help='the directory that used to save logs')
parser.add_argument('--train-dir', type=str, 
		default='hdfs://yz-cpu-vm001.hogpu.cc:8020/user/great_searcher/train/',
		help='the input train data directory')
parser.add_argument('--cv-dir', type=str, 
		default='hdfs://yz-cpu-vm001.hogpu.cc:8020/user/great_searcher/cv/',
		help='the input validation data directory')
parser.add_argument('--model-dir', type=str, 
		default='hdfs://yz-cpu-vm001.hogpu.cc:8020/user/great_searcher/models/',
		help='the directory that used to save models')

parser.add_argument('--gpus', type=str, default='0',
		help='the gpus that will be used')
parser.add_argument('--num-examples', type=str, default='40000000',
		help='the number of examples')
parser.add_argument('--num-classes', type=str, default='2',
		help='the label`s number of classes')
parser.add_argument('--num-epoch', type=str, default='10',
		help='the number of epoch')
parser.add_argument('--batch-size', type=str, default='256',
		help='the batch size for each train')
args = parser.parse_args()

import subprocess
import time
import copy
import os
test_cnt = 0  #the begining number of logs, such as 0.log, 1.log and 2.log
lunch_time = time.time() #time stamp, used to avoid filename collision
lr_factor = "0.1"
lr_factor_epoch = "3,8"

def lunch_test(lr_factor = 1, lr_factor_epoch = 1):
	write_config()
	log = path_prefix + "/log/" + str(test_cnt) + ".log"
	command = ("python train_speech.py --model-dir " + args.model_dir+ str(lunch_time) + '_' + str(test_cnt) + 
			" --train-dir " + args.train_dir +
			" --cv-dir " + args.cv_dir +
			" --gpus " + args.gpus +
			" --num-epochs " + args.num_epoch +
			" --num-examples " + args.num_examples +
			" --num-classes " + args.num_classes +
			" --lr-factor " + str(lr_factor) +
			" --lr-factor-epoch " + str(lr_factor_epoch) +
			" --batch-size " + args.batch_size +
			" 2>&1 ")
	child = subprocess.Popen(command, shell  = True)
	child.wait()

if "__main__" == __name__:
	net_index = open("./configs/net_configs/index").readlines()
	opt_index = open("./configs/opt_configs/index").readlines()

	assert len(net_index) == len(opt_index), ("the number of entries for net_index (value:" + str(len(net_index))
			+ ") doesn`t equal the number of entries for opt_index (value:" + str(len(opt_index)) + ")" )

	for val in zip(net_index, opt_index):
		net_filename = val[0].strip()
		opt_filename = val[1].strip()


