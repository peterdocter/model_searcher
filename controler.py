import argparse
parser = argparse.ArgumentParser(description='train a speech classifer')
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
import mxnet as mx
from symbol_factory import symbol_factory as _sf
from optimizer_factory import optimizer_factory as _of
test_cnt	= 0  #the begining number of logs, such as 0.log, 1.log and 2.log
lunch_time	= time.time() #time stamp, used to avoid filename collision
lr_factor	= "0.1"
lr_factor_epoch = "3,8"

def lunch_test(lr_factor = "1", lr_factor_epoch = "1"):
	print "starting training, test_id =", test_cnt
	command = ("python train_speech.py" + 
			" --model-dir "		+ args.model_dir + '/' + str(lunch_time) + '_' + str(test_cnt) + "/ "
			" --train-dir "		+ args.train_dir	+
			" --cv-dir "		+ args.cv_dir		+
			" --gpus "			+ args.gpus			+
			" --num-epochs "	+ args.num_epoch	+
			" --num-examples "	+ args.num_examples +
			" --num-classes "	+ args.num_classes  +
			" --lr-factor "		+ lr_factor			+
			" --lr-factor-epoch " + lr_factor_epoch +
			" --batch-size "	+ args.batch_size	+
			" 2>&1 ")
	child = subprocess.Popen(command, shell = True)
	child.wait()

def file_format_check (net_index, opt_index):
	print "checking configs"
	assert len(net_index) == len(opt_index), ("the number of entries for net_index (value:" + str(len(net_index))
			+ ") doesn`t equal the number of entries for opt_index (value:" + str(len(opt_index)) + ")" )

	line = 1
	for val in zip(net_index, opt_index):
		net_filename = val[0].strip()
		opt_filename = val[1].strip()
		try:
			net_config = open("./configs/net_configs/" + net_filename).read()
		except:
			raise Exception("file" + net_filename + "doesn`t exist, net_config index file line number = " + line)

		try:
			opt_config = open("./configs/opt_configs/" + opt_filename).read()
		except:
			raise Exception("file" + opt_filename + "doesn`t exist, opt_config index file line number = " + line)

		if "lr_factor" in opt_config:
			lr_factor_index 	  = opt_config.find("lr_factor=")
			lr_factor_epoch_index = opt_config.find("lr_factor_epoch=")

			assert lr_factor_epoch_index != -1, ("did you forget parameter lr_factor_epoch? file_name = " 
					+ opt_filename)
			assert lr_factor_epoch != -1, ("did you forget parameter lr_factor? file_name = " 
					+ opt_filename)

			contain	   = opt_config.split()
			opt_config = ""
			for part in contain:
				if "lr_factor" not in part:
					opt_config += part 
					opt_config += " "
		print "testing net-opt pair : " + str(line)
		sf = _sf()
		of = _of()
		try:
			net = sf(net_config)
			net.infer_shape(data=(int(args.batch_size), 11, 3, 40), softmax_label = (int(args.batch_size),))
		except:
			raise Exception("compile file \"" + net_filename + "\" failed, parameter name error?")
		try:
			opt_name, opt_params = of(opt_config)
			mx.optimizer.create(opt_name, **opt_params)
		except:
			raise Exception("compile file \"" + opt_filename + "\" failed, parameter name error?")
		line += 1
	print "check over, format check all passed"
	print ""
	print ""
	print ""

if "__main__" == __name__:
	net_index = open("./configs/net_configs/index").readlines()
	opt_index = open("./configs/opt_configs/index").readlines()

	file_format_check(net_index, opt_index)
	
	for val in zip(net_index, opt_index):
		net_filename = val[0].strip()
		opt_filename = val[1].strip()
		net_config = open("./configs/net_configs/" + net_filename).read()
		opt_config = open("./configs/opt_configs/" + opt_filename).read()

		if "lr_factor" in opt_config:
			lr_factor_index 	  = opt_config.find("lr_factor=")
			lr_factor_epoch_index = opt_config.find("lr_factor_epoch=")

			contain	   = opt_config.split()
			opt_config = ""
			for part in contain:
				if "lr_factor" not in part:
					opt_config += part 
					opt_config += " "
				else:
					param_val = part.split("=")
					if param_val[0] == "lr_factor":
						lr_factor = param_val[1]
					else:
						lr_factor_epoch = param_val[1]
					
		net_file = open(".net_config", "w").write(net_config)
		opt_file = open(".opt_config", "w").write(opt_config)

		lunch_test(lr_factor, lr_factor_epoch)
		test_cnt += 1
