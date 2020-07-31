import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import pickle
from subset_helper import bax
from function_factories_frontier import sum_log_prob_weighted_factory, sum_prob_weighted_factory,sum_distance_weighted_factory,mean_distance_function,mean_probability_function,type_covariance_function, price_of_fairness_distance_function, price_of_fairness_probability_function
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True
import sys

class Facility:
	def __init__(self,location=None,maxCoord=None,*args,**kwargs):
		if location is not None:
			self.location = location
		elif maxCoord is not None:
			self.location = [np.random.uniform(0,maxCoord) for i in range(2)]
		else:
			self.location = None
	def dist(self,other_thing):
		return (self.location[0] - other_thing.location[0])**2 + (self.location[1] - other_thing.location[1])**2

class Person(Facility):
	def __init__(self,ttheta, location=None,*args,**kwargs):
		super().__init__(location,*args,**kwargs)
		self.ttheta = ttheta

# Person(theta[i],maxCoord = sizeRegion)


nIndiv = 50
nFac = 50
nSelectedFac = 3
nTrials = 1

sizeRegion = 1


theta_low = 0
theta_high = 1
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
nIndiv = len(theta)
np.random.shuffle(theta)
theta_low_indices = [i for i in range(nIndiv) if theta[i]==theta_low]

beta = [0,0.5,-10] #[beta0, beta_theta >0, beta_r <0]

weights = [1,5,1000] #should all be >= 1
log_prob_obj_fns = [sum_log_prob_weighted_factory(w) for w in weights]
prob_obj_fns = [sum_prob_weighted_factory(w) for w in weights]
dist_obj_fns= [sum_distance_weighted_factory(w) for w in weights]
objective_functions = log_prob_obj_fns + prob_obj_fns + dist_obj_fns
SWB_dist = dist_obj_fns[0]
SWB_prob = prob_obj_fns[0]
SWB_functions = [SWB_dist, SWB_prob]

indices = [
	{'name':'Mean Distance','key': 'mean_dist', 'shared':False,'fn':mean_distance_function},
	{'name':'Mean Success Prob','key': 'mean_prob','shared':False,'fn':mean_probability_function},
	{'name':'Covariance(type,success)','key': 'cov','shared':True,'fn':type_covariance_function},{'name':'Price of Fairness: Distance','key': 'pof_dist','shared':True,'fn':price_of_fairness_distance_function},
	{'name':'Price of Fairness: Probability','key': 'pof_prob','shared':True,'fn':price_of_fairness_probability_function}
	]


objective_function_names = {fn.__name__ : fn for fn in objective_functions}

fairness_ratings = [10**10 + weight for weight in weights]+ [10**5 + weight for weight in weights] + weights #arbitrary, but roughly increasing with fairness-inducing tendency
fairness_strengths = dict(zip(objective_functions,fairness_ratings)) # to create a consistent order in plot legends

def add_dict_lists(dict1,dict2):
		return {k: np.concatenate((dict1[k], dict2[k])) for k in dict1}
def size_of_dict_lists(d):
	return sum([sys.getsizeof(l) for k,l in d.items()])
class Instance:
	## TODO:
	# make all "fn" keys take a uniform object and find their own appropriate values
	def __init__(self, rvals=None, yvals=None, uvals=None, fstars=None, uvals_type0=None,fstars_type0=None): #, facilities=None, individuals=None
		self.rvals = rvals
		self.yvals = yvals
		self.theta = theta
		self.theta_high = theta_high
		self.theta_low = theta_low
		self.beta = beta
		if uvals is not None:
			self.uvals = uvals
		else:
			self.uvals = {fn: fn(self) for fn in objective_functions}
		if fstars is not None:
			self.fstars = fstars
		else:
			self.fstars = {fn: sum(self.uvals[fn]) for fn in objective_functions}
		if uvals_type0 is not None:
			self.uvals_type0 = uvals_type0
		else:
			self.uvals_type0 = {fn: self.uvals[fn][theta_low_indices] for fn in objective_functions}
		if fstars_type0 is not None:
			self.fstars_type0 = fstars_type0
		else:
			self.fstars_type0 = {fn: sum(self.uvals_type0[fn]) for fn in objective_functions}


	def combine(self, other_instance):
		if other_instance is not None:
			# print(f"len(self.rvals: {len(self.rvals)}, len(other_instance.rvals): {len(other_instance.rvals)}")
			return Instance(
			                rvals = self.rvals + other_instance.rvals,
			                yvals = self.yvals + other_instance.yvals,
			                uvals = add_dict_lists(self.uvals,other_instance.uvals),
			                fstars = {fn: sum(self.uvals[fn]) for fn in objective_functions},
			                uvals_type0 = add_dict_lists(self.uvals_type0,other_instance.uvals_type0),
			                fstars_type0 = {fn: sum(self.uvals_type0[fn]) for fn in objective_functions}
			                )
		else:
			return self
	def size(self):
		# print()
		# print(f"rvals: {sys.getsizeof(self.rvals)} + yvals: {sys.getsizeof(self.yvals)} + uvals: {size_of_dict_lists(self.uvals)} + uvals_type0: {size_of_dict_lists(self.uvals_type0)} + fstars: {size_of_dict_lists(self.fstars)} + fstars_type0: {size_of_dict_lists(self.fstars_type0)}")
		return sys.getsizeof(self.rvals) + sys.getsizeof(self.yvals) + size_of_dict_lists(self.uvals) + size_of_dict_lists(self.uvals_type0) + size_of_dict_lists(self.fstars) + size_of_dict_lists(self.fstars_type0)

	def beats(self,other_instance,fn):
		if other_instance is not None:
			return self.fstars[fn] > other_instance.fstars[fn]
		else:
			return True

	def dominates(self,other_instance,fn):
		return self.fstars[fn] > other_instance.fstars[fn] and self.fstars_type0[fn] > other_instance.fstars_type0[fn]

	def dominators(self,other_instance):
		return {fn: (self.fstars[fn] > other_instance.fstars[fn] and self.fstars_type0[fn] > other_instance.fstars_type0[fn]) for fn in objective_functions}

class Frontier:
	def __init__(self,fn):
		self.instances = []
		self.fn = fn
	def dominates(self,other_instance):
		for instance in self.instances:
			if instance.dominates(other_instance,self.fn):
				return True
		return False
	def append(self, other_instance):
		self.instances = [instance for instance in self.instances if not other_instance.dominates(instance,self.fn)]
		self.instances.append(other_instance)
	def size(self):
		return sum([instance.size() for instance in self.instances])
	def fstarvals(self):
		return [instance.fstars[self.fn] for instance in self.instances]
	def fstarvals_type0(self):
		return [instance.fstars_type0[self.fn] for instance in self.instances]
# TODO: merge frontiers
def average_frontiers(self,list_of_frontiers):
	pass

def solve_multiple_frontier(saving=False):
	bbax=bax()
	frontiers = {fn: [Frontier(fn) for trial in range(nTrials)] for fn in SWB_functions}
	all_best = {fn: None for fn in objective_functions}

	for trial in range(nTrials):
		best = {fn: None for fn in objective_functions}
		indiv = [Person(theta[i],maxCoord = sizeRegion) for i in range(nIndiv)]
		fac = [Facility(maxCoord = sizeRegion) for i in range(nFac)]
		dist = np.array([[person.dist(facility) for facility in fac] for person in indiv])
		for ind, gp in enumerate(bbax.bax_gen(nFac,nSelectedFac)):
			gp = gp.toList()
			r = [min((dist[i][j] for j in gp)) for i in range(nIndiv)]
			# print(len(r))
			yv = np.zeros((nFac),dtype=int)
			yv[gp] = 1
			instance = Instance(rvals=r, yvals=yv)
			for fn in frontiers:
				frontier = frontiers[fn][trial]
				if not frontier.dominates(instance):
					frontier.append(instance)
			# for fn, best_instance in best.items():
			# 	if instance.beats(best_instance,fn):
			# 		best[fn] = instance
			best = {fn : instance if instance.beats(best_instance,fn) else best_instance for fn,best_instance in best.items()}
		for fn,combined_instance in all_best.items():
			all_best[fn] = best[fn].combine(combined_instance)
		print(f"Solved {trial+1}/{nTrials} enumeratively")
	# if saving:
	# 	save_data(dat)
	return all_best,frontiers
	# return all_best


def plot_frontier(list_of_frontiers):
	fig, ax = plt.subplots(figsize=(10,10))
	# plt.axis('off')
	title = f"Efficient Frontier for Maximizing Distance"
	plt.title(title)
	ax.set_xlabel("Utility to $\\theta_0=0$ Individuals")
	ax.set_ylabel("Utility to Entire Population")
	# number_points = np.lcm.reduce([len(front.instances) for front in list_of_frontiers])
	all_xvals = sorted(set((xval for front in list_of_frontiers for xval in front.fstarvals_type0())))
	x_raw = [front.fstarvals_type0() for front in list_of_frontiers]
	y_raw = [front.fstarvals() for front in list_of_frontiers]
	#discard trials where one selection was superior for both groups
	x_raw = [vals for vals in x_raw if len(vals) > 1]
	y_raw = [vals for vals in y_raw if len(vals) > 1]
	nTrialsUsed = len(x_raw)
	y_interp = (1/nTrialsUsed) * sum([np.interp(all_xvals,x_raw[i],y_raw[i]) for i in range(nTrialsUsed)])
	plt.scatter(all_xvals,y_interp)
	# plt.scatter(frontier.fstarvals(),frontier.fstarvals_type0())
	# leg = ax.legend(loc="best",prop={"size":8})
	plt.show(block=False)
	return fig, ax, all_xvals,y_interp


all_best,frontiers = solve_multiple_frontier()
print([len(front.instances) for front in frontiers[SWB_dist]])
frontier = frontiers[SWB_dist][np.argmax([len(front.instances) for front in frontiers[SWB_dist]])]
# fig,ax,xx,yy = plot_frontier(frontiers[SWB_dist])


list_of_frontiers = frontiers[SWB_dist]
fig, ax = plt.subplots(figsize=(10,10))# plt.axis('off')
title = f"Efficient Frontier for Maximizing Distance"
plt.title(title)
ax.set_xlabel("Utility to $\\theta_0=0$ Individuals")
ax.set_ylabel("Utility to Entire Population")
x_raw = [front.fstarvals_type0() for front in list_of_frontiers]
y_raw = [front.fstarvals() for front in list_of_frontiers]
#discard trials where one selection was superior for both groups
min_points = 5
x_raw = [vals for vals in x_raw if len(vals) > min_points]
y_raw = [vals for vals in y_raw if len(vals) > min_points]
nTrialsUsed = len(x_raw)
x_percentage = [np.array([xval/yval for (xval,yval) in zip(x_raw[i],y_raw[i])]) for i in range(nTrialsUsed)] #percentage of total utility enjoyed by type0 individuals
all_xvals = sorted(set((xval for front in x_percentage for xval in front )))
y_interp = (1/nTrialsUsed) * sum([np.interp(all_xvals,x_percentage[i],y_raw[i]) for i in range(nTrialsUsed)])
plt.scatter(all_xvals,y_interp)
plt.show(block=False)
