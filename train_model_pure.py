import mxnet as mx
import data_loader

#net is the symbol of the network
#opt_name is the name(type:str) of the optimizer which will be used
#opt_name is a dict that stores the aaddition parameters of the optimizer
def fit(args, network, opt_name, opt_params):
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

	# save model?
	checkpoint = mx.callback.do_checkpoint(args.model_prefix)

	# data
	(train, val) = data_loader.data_loader(args, kv)

	# train
	devs = mx.cpu() if args.gpus is None else [
			mx.gpu(int(i)) for i in args.gpus.split(',')]

	#calculate the epoch size
	epoch_size = args.num_examples / args.batch_size
	if args.kv_store == 'dist_sync':
		epoch_size /= kv.num_workers
		model_args['epoch_size'] = epoch_size

	#create lr_scheduler for SGD
	if args.lr_factor < 1 and "SGD" == opt_name:
		step_epochs = [int(l) for l in args.lr_factor_epoch.split(',')]
		_step = [epoch_size * (x-args.load_epoch) for x in step_epochs if x-args.load_epoch > 0]
		opt_params['lr_scheduler'] = mx.lr_scheduler.MultiFactorScheduler(
				step   = _step,
				factor = args.lr_factor)
		model_args["lr_scheduler"] = opt_params["lr_scheduler"]
		print ">>>> Add lr_scheduler to sgd"
		print ">>>> lr_factor =", args.lr_factor, ", lr_factor_epoch =", args.lr_factor_epoch

	# set clip_gradient for optimizer
	if 'clip_gradient' in args and args.clip_gradient is not None:
		model_args['clip_gradient'] = args.clip_gradient
		opt_params['clip_gradient'] = model_args['clip_gradient']

	# disable kvstore for single device
	if 'local' in kv.type and (
			args.gpus is None or len(args.gpus.split(',')) is 1):
		kv = None

	#for hadoop use
	if args.hybrid_num_batches_per_block:
		#assert args.kv_store == 'dist_sync'
		batch_size_per_device = args.batch_size / len(devs)
		num_devices_all = len(devs) * kv.num_workers if args.kv_store == 'dist_sync' else len(devs)
		local_optimizer = mx.optimizer.create('sgd', learning_rate = args.lr, momentum = 0.0, 
				rescale_grad=(1.0/batch_size_per_device), **opt_params)
		remote_optimizer = mx.optimizer.create('bmuf', learning_rate = 1.0, momentum = 0.9, is_nbm=False, 
				rescale_grad=(1.0/num_devices_all))
		opt_name = [local_optimizer, remote_optimizer]

	#make module
	model = mx.model.FeedForward(
			ctx                = devs,
			symbol             = network,
			num_epoch          = args.num_epochs,
			learning_rate      = opt_params["learning_rate"],
			momentum           = opt_params["momentum"],
			wd                 = opt_params["wd"],
			optimizer          = opt_name,
#			initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
			**model_args)

	#start training
	model.fit(
			X                  = train,
			eval_data          = val,
			kvstore            = kv,
			batch_end_callback = mx.callback.Speedometer(args.batch_size, 1000),
			epoch_end_callback = checkpoint)
