import mxnet as mx
class optimizer_factory (object):
	def __init__ (self):
		self.optimizer_pool = {}
		self.optimizer_pool["adad"]   = "AdaDelta"
		self.optimizer_pool["adag"]   = "AdaGrad"
		self.optimizer_pool["adam"]   = "Adam"
		self.optimizer_pool["dcasgd"] = "DCASGD"
		self.optimizer_pool["nag"]    = "NAG"
		self.optimizer_pool["rms"]    = "RMSProp"
		self.optimizer_pool["sgd"]    = "SGD"
		self.optimizer_pool["sgld"]   = "SGLD"

		self.config_pool = {}
		self.config_pool["adad"]   = {
				"rho"     : 0.01,
#				"epsilon" : 1e-10,
#				"wd"      : 0.0001
				}
		self.config_pool["adag"]   = {
				"learning_rate" : 0.01,
#				"epsilon"		: 1e-10,
#				"wd"			: 0.0001
				}
		self.config_pool["adam"]   = {
				"learning_rate" : 0.0001,
#				"epsilon"		: 1e-8,
#				"wd"			: 0.0001,
#				"beta1"         : 0.9,
#				"beta2"         : 0.999
				}
		self.config_pool["dcasgd"] = {
				"learning_rate" : 0.05,
				"momentum"      : 0.01,
#				"wd"            : 0.0001,
#				"lamda"         : 1
				}
		self.config_pool["nag"]    = {
				"learning_rate" : 0.05,
				"momentum"      : 0.01,
#				"wd"            : 0.0001
				}
		self.config_pool["rms"]    = {
				"learning_rate" : 0.0001,
#				"gamma1"        : 0.9,
#				"gamma2"        : 0.9,
#				"centered"      : True,
#				"wd"            : 0.0001,
#				"epsilon"       : 1e-10
				}
		self.config_pool["sgd"]    = {
				"learning_rate" : 0.05,
				"momentum"      : 0.01,
#				"wd"            : 0.0001
				}
		self.config_pool["sgld"]   = {
				"learning_rate" : 0.01,
#				"wd"            : 0.0001
				}

	def __call__ (self, cmds):
		print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
		self.decode(cmds)
		print self.realname, self.params
		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
		return self.realname, self.params

	def check_params (self, name, params):
		config = self.config_pool[name]
		for key in config:
			if not params.get(key, False):
				params[key] = config[key]
				print "warning :", name, "didn`t get parameter", key
				print "          use default", config[key]

		return params

	def decode (self, cmd):
		cmd = cmd.strip(" \n")
		name = cmd.split(' ', 1)[0]
		args = ""
		try:
			args = cmd.split(' ')[1:]
		except:
			pass

		kw = {}
		for key_val in args:
			key = key_val.split('=')[0]
			val = None
			try:
				val = key_val.split('=')[1]
			except:
				print  "did you forget to write \"=\"?"
				raise Exception()

			if "True" == val or "False" == val:
				val = bool(val)
			try:
				val = float(val)
			except:
				raise Exception("Error : can not convert", val, "to float")

			kw[key] = val

		self.realname = self.optimizer_pool[name]
		self.params   = self.check_params(name, kw)

if "__main__" == __name__:
	of = optimizer_factory()
	print of("rms   ")

