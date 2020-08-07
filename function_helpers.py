def setmeta(**kwargs):
	def namer(f):
		for k,v in kwargs.items():
			setattr(f,k,v)
		return f
	return namer
