import mxnet as mx
def data_loader(args, kv):
	train = mx.io.SpeechRecordIter(
			path_rec				= args.train_dir,
			data_shape				= (40,),
			num_classes				= args.num_classes,
			context_size			= 5,
			batch_size				= args.batch_size,
			num_parts				= kv.num_workers,
			part_index              = kv.rank,
			num_sentences_per_chunk = 20000,
			mean_var_file			= args.meanvar_file,
			shuffle					= True)

	cv = mx.io.SpeechRecordIter(
			path_rec				= args.cv_dir,
			data_shape				= (40,),
			num_classes				= args.num_classes,
			context_size			= 5,
			batch_size				= args.batch_size,
			num_parts				= kv.num_workers,
			part_index				= kv.rank,
			num_sentences_per_chunk = 10000,
			mean_var_file			= args.meanvar_file,
			shuffle					= False)
	return (train, cv)
