import mxnet as mx
class symbol_factory (object):
	def __init__ (self):
		self.layer_pool = {}
		self.layer_pool["conv"] = mx.symbol.Convolution
		self.layer_pool["pool"] = mx.symbol.Pooling
		self.layer_pool["flat"] = mx.symbol.Flatten
		self.layer_pool["norm"] = mx.symbol.BatchNorm
		self.layer_pool["act"]  = mx.symbol.Activation
		self.layer_pool["out"]  = mx.symbol.SoftmaxOutput
		self.layer_pool["fc"]   = mx.symbol.FullyConnected

		self.config_pool = {}
		self.config_pool["conv"] = {
				"kernel"	 : (11, 9),
				"num_filter" : 128
				}
		self.config_pool["pool"] = {
				"kernel"	 : (1, 4),
				"stride"     : (1, 4),
				"pool_type"  : "max"
				}
		self.config_pool["flat"] = {}
		self.config_pool["norm"] = {}
		self.config_pool["act"]  = {
				"act_type"   : "relu"
				}
		self.config_pool["out"]  = {
				"name"       : "softmax"
				}
		self.config_pool["fc"]   = {
				"num_hidden" : 1024
				}

	def __call__ (self, cmds):
		self.decode (cmds)
		print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
		self.make_symbol()
		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
		return self.output_symbol

	def check_params (self, layername, params):
		config = self.config_pool[layername]
		for key in config:
			if not params.get(key, False):
				params[key] = config[key]
				print "warning :", layername, "didn`t get parameter", key
				print "          use default", config[key]

		return params

	def decode (self, cmds):
		conv_cmds = [[], []]
		cmds = cmds.split('\n')
		for line in cmds:
			line = line.strip()
			if "" == line:
				break
			name = line.split(' ', 1)[0]
			args = []
			try:
				args = line.split(' ')[1:]
			except:
				pass

			kw = {}
			for key_val in args:
				if "" == key_val:
					break
				key = key_val.split('=')[0]
				val = None
				try:
					val = key_val.split('=')[1]
				except:
					print  "did you forget to write \"=\"?"
					raise Exception()
				kw[key] = val

			conv_cmds[0].append(name)
			conv_cmds[1].append(kw)

		self.conv_cmds = conv_cmds

	def make_symbol(self):
		conv_cmds = self.conv_cmds
		self.conv_cmds = None

		tmp_symbol = mx.symbol.Variable(name="data")
		tmp_symbol = mx.symbol.Reshape(data=tmp_symbol, target_shape=(0, 11, 3, 40))
		tmp_symbol = mx.symbol.NHCW2Conv(data=tmp_symbol, dim1=1, dim2=2)

		length = len(conv_cmds[0])
		cnt = 0
		while cnt != length:
			func = self.layer_pool[conv_cmds[0][cnt]]
			kw = conv_cmds[1][cnt]
			kw = self.check_params(conv_cmds[0][cnt], kw)
			
			print "layer", cnt, ":", conv_cmds[0][cnt], conv_cmds[1][cnt] #for debug
			tmp_symbol = func(data = tmp_symbol, **kw)
			cnt += 1

		self.output_symbol = tmp_symbol

		self.output_symbol.save("./debug.json")

if "__main__" == __name__:
	sf = symbol_factory()
	sf("conv kernel=(1,8)\npool pool_type=avg\nflat\nnorm\nact\nfc\nnorm\nact\nfc\nout")
