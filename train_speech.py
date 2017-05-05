import argparse
import time
import os
import logging
import subprocess
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='train a speech classifer')
parser.add_argument('--train-dir', type=str, default='train/',
                    help='the input train data directory')
parser.add_argument('--cv-dir', type=str, default='cv/',
                    help='the input validation data directory')
parser.add_argument('--model-dir', type=str, default="models/",
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--meanvar-file', type=str, default='meanvar_ndarray',
                    help='the mean variance file of train data')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=1e9,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
					help='the batch size')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int, default=0,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--num-classes', type=int, default=2,
                    help='the num of classes of the data')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=str, default="1",
                    help='the number of epoch to factor the lr')
parser.add_argument('--hybrid-num-batches-per-block', type=int, default=None,
					help='enable hybrid kvstore, and set num batches per block')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
args = parser.parse_args()
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from symbol_factory import symbol_factory as _sf
from optimizer_factory import optimizer_factory as _of
sf = _sf()
of = _of()

assert args.opt_config_name != "null"
assert args.net_config_name != "null"	

#check dirs
try:
	if "hdfs://" in args.model_dir:
		child = subprocess.Popen("hdfs dfs -mkdir " + args.model_dir,shell = True)
		return_code = child.wait()
		if return_code != 0:
			raise Exception("hdfs model dir exist, abort")
	else:
		os.mkdirs(args.model_dir)
except:
	raise Exception("local model dir exist, abort")

#main code
net_cmds = open("./.net_config").read()
opt_cmds = open("./.opt_config").read()
net = sf(net_cmds)
opt_name, opt_params = of(opt_cmds)

import train_model as train_model
train_model.fit(args, net, opt_name, opt_params)
