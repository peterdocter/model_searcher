1.please change the path in "remote_2831" and "remote_vad"
2.make sure that each net_config in file "index" has a corresponding opt_config in file "index"
3.put net_configs under "./config/net_configs/". make sure you have added it into the index file,
	or it won`t be used
4.same file can be used for multiple times, just write it multiple times in "index"

[Example format for net index]
net_a
net_b
net_b
net_c

[Example format for opt index]
opt_a
opt_b
opt_a
opt_b

Notice:	the net in line 1 (as the name is net_a) will be loaded with the opt in line 1(as the name is opt_a)
		the net in line 2 (as the name is net_b) will be loaded with the opt in line 2(as the name is opt_b)
		the net in line 3 (as the name is net_b) will be loaded with the opt in line 3(as the name is opt_a)
		respective.

[Example format for net_config]
conv kernel=(1,1) stride=(1,1) pad=(0,0) num_filter=128
pool kernel=(1,1) stride=(1,1) pad=(0,0) pool_type=max
flat
norm
act act_type=relu
fc num_hidden=1024
out name=softmax

Notice:	one layer per line, DO NOT use space before and after the equality sign.
		you can also add other parameters
 
[Example format for net_config]
sgd learning_rate=0.02 momentum=0.9 lr_factor=0.1 lr_factor_epoch=2,4,6,8

Notice:	write all the thing in one line, DO NOT use space before and after the equality sign.
		you can also add other parameters. Available optimizers are adam, rms, sgd and so on. 
		parameter lr_factor only useful for optimizer sgd
