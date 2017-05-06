import mxnet as mx
import logging
from data_loader import data_loader
#net is the symbol of the network
#opt_name is the name(type:str) of the optimizer which will be used
#opt_name is a dict that stores the aaddition parameters of the optimizer

def fit(args, net, opt_name, opt_params):
	logging.info('start with arguments %s', args)
	# kvstore
	kv = mx.kvstore.create(args.kv_store,
			hybrid_num_batches_per_block=args.hybrid_num_batches_per_block)

	# load model?
	model_args = {}
	if args.load_epoch > 0:
		tmp = mx.model.FeedForward.load(args.model_prefix, args.load_epoch)
		model_args = {
				'arg_params'  : tmp.arg_params,
				'aux_params'  : tmp.aux_params,
				'begin_epoch' : args.load_epoch}

	#load data
	train, val = data_loader(args, kv)

	#calculate dev context
	devs = mx.cpu()
	try:
		devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]
	except:
		pass

	#conculate epoch_size
	epoch_size = args.num_examples / args.batch_size
	if args.kv_store == 'dist_sync':
		epoch_size /= kv.num_workers

	#create lr_scheduler for SGD
	if args.lr_factor < 1 and "SGD" == opt_name:
		step_epochs = [int(l) for l in args.lr_factor_epoch.split(',')]
		_step = [epoch_size * (x-args.load_epoch) for x in step_epochs if x-args.load_epoch > 0]
		opt_params['lr_scheduler'] = mx.lr_scheduler.MultiFactorScheduler(
				step   = _step,
				factor = args.lr_factor)
		print ">>>> Add lr_scheduler to sgd"
		print ">>>> lr_factor =", args.lr_factor, ", lr_factor_epoch =", args.lr_factor_epoch

	# disable kvstore for single device
	if "local" in kv.type and (args.gpus == None or len(args.gpus.split(',')) is 1):
		kv = None

	#create model
	model = mx.mod.Module(
			context   = devs,
			symbol    = net)

	#train model
	model.fit(
			train_data				= train,
			eval_data				= val,
			num_epoch				= args.num_epochs,
			optimizer				= opt_name,
			optimizer_params		= opt_params,
			initializer				= mx.init.Xavier(factor_type="in", magnitude=2.34),
			kvstore					= kv,
			batch_end_callback		= mx.callback.Speedometer(args.batch_size, 1000),
			epoch_end_callback		= mx.callback.do_checkpoint(args.model_dir),
			**model_args)


