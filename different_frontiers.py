import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import pickle
from subset_helper import bax
from function_factories_frontier import sum_log_prob_weighted_factory, sum_prob_weighted_factory,sum_distance_weighted_factory,mean_distance_function,mean_probability_function,type_covariance_function, price_of_fairness_distance_function, price_of_fairness_probability_function,markers,covariance_factory, linear_combination_of_objectives
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
nFac = 20
nSelectedFac = 4

##TESTING
# nIndiv = 40
# nFac = 20
# nSelectedFac = 4



sizeRegion = 1


theta_low = 0
theta_high = 1
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
nIndiv = len(theta)
# np.random.shuffle(theta)
theta_low_indices = [i for i in range(nIndiv) if theta[i]==theta_low]

beta = [0,1,-10] #[beta0, beta_theta >0, beta_r <0]

weights = [0,0.8] #should all be >= 1
log_prob_obj_fns = [sum_log_prob_weighted_factory(w) for w in weights]
prob_obj_fns = [sum_prob_weighted_factory(w) for w in weights]
dist_obj_fns= [sum_distance_weighted_factory(w) for w in weights]
SWB_dist = dist_obj_fns[0]
SWB_prob = prob_obj_fns[0]
SWB_logprob = log_prob_obj_fns[0]
SWB_functions = [SWB_dist,SWB_prob,SWB_logprob]#
cov_fn = covariance_factory()
prob_plus_lambda_cov = [linear_combination_of_objectives(SWB_prob,cov_fn,w) for w in weights]

objective_functions = log_prob_obj_fns + prob_obj_fns + dist_obj_fns + prob_plus_lambda_cov

indices = [
	# {'name':'Mean Distance','key': 'mean_dist', 'shared':False,'fn':mean_distance_function},
	# {'name':'Mean Success Prob','key': 'mean_prob','shared':False,'fn':mean_probability_function},
	# {'name':'Covariance(type,success)','key': 'cov','shared':True,'fn':type_covariance_function},{'name':'Price of Fairness: Distance','key': 'pof_dist','shared':True,'fn':price_of_fairness_distance_function},
	# {'name':'Price of Fairness: Probability','key': 'pof_prob','shared':True,'fn':price_of_fairness_probability_function}
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
	def sort_key(self,fn):
		return self.fstars[fn]

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
		self.sort()
	def sort(self):
		self.instances.sort(key = lambda instance: instance.sort_key(self.fn))
	def size(self):
		return sum([instance.size() for instance in self.instances])
	def fstarvals(self,fn=None):
		if fn is None:
			fn = self.fn
		return [instance.fstars[fn] for instance in self.instances]
	def fstarvals_type0(self,fn=None):
		if fn is None:
			fn = self.fn
		return [instance.fstars_type0[fn] for instance in self.instances]
	def consider(self,other_instance):
		if not self.dominates(other_instance):
			self.append(other_instance)
# TODO: merge frontiers
def average_frontiers(self,list_of_frontiers):
	pass

def solve_multiple_frontier(saving=False):
	bbax=bax()
	frontiers = {fn: Frontier(fn) for fn in SWB_functions}
	best = {fn: None for fn in objective_functions}
	indiv = [Person(theta[i],maxCoord = sizeRegion) for i in range(nIndiv)]
	fac = [Facility(maxCoord = sizeRegion) for i in range(nFac)]
	# dist = np.array([[person.dist(facility) for facility in fac] for person in indiv])
	dist = pickle.load(open('last_dist_frontier_one_trial_10pts.pickle','rb'))
	total_subsets = bbax.enumerator.choose(nFac,nSelectedFac)
	for ind, gp in enumerate(bbax.bax_gen(nFac,nSelectedFac)):
		if ind % 1000 == 0:
			num_in_each_frontier = [len(frontier.instances) for fn, frontier in frontiers.items()]
			print(f"Processed {ind} subsets out of {total_subsets}")
			print(num_in_each_frontier)
		gp = gp.toList()
		r = [min((dist[i][j] for j in gp)) for i in range(nIndiv)]
		# print(len(r))
		yv = np.zeros((nFac),dtype=int)
		yv[gp] = 1
		instance = Instance(rvals=r, yvals=yv)
		for fn,frontier in frontiers.items():
			frontier.consider(instance)
		best = {fn : instance if instance.beats(best_instance,fn) else best_instance for fn,best_instance in best.items()}
	# if saving:
	#   save_data(dat)
	return best,frontiers,dist

def generate_file_label():
	weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_label = f"at_{timestamp}_weights_{weights_label}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nSelectedFac}of{nFac}"
	return trial_label
def get_n_colors(number_desired_colors):
	if number_desired_colors < 10:
		default = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		colors = default[0:number_desired_colors]
	else:
		# https://matplotlib.org/tutorials/colors/colormaps.ht
		cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
		colors = [cmap(i) for i in range(number_desired_colors)]
	return colors



def plot_tradeoffs(best,frontiers,saving=False):
	# fig, ax = plt.subplots(figsize=(10,10))
	# fig = plt.figure(figsize=(10,10))
	axes = {}
	colors = get_n_colors(len(best))
	color_map = {fn : colors.pop() for fn in best}
	# marker_map = {fn : f"${fn.marker}^{{{fn.weight}}}$" for fn in best}
	marker_map = {fn : f"{fn.mpl_marker}" for fn in best}
	# marker_map = {fn : fn.weighted_marker for fn in best}
	# fake_marker_map = {fn : f"${fn.marker}^{{{fn.coeff}}}$" for fn in best}
	line_artists = []
	nPlots = len(frontiers)
	gs = gridspec.GridSpec(nrows=nPlots,ncols=1,height_ratios=np.ones(nPlots))
	fig = plt.figure(figsize=(5,5*nPlots))
	fig.subplots_adjust(hspace=0.14)


	for i, (plot_fn,plot_frontier) in enumerate(frontiers.items()):
		ax = plt.subplot(gs[i])
		ax.set_xticks([])
		ax.set_yticks([])
		line_artists = []
		for (fn,frontier) in frontiers.items():
			x_raw = frontier.fstarvals_type0(fn=plot_fn)
			y_raw = frontier.fstarvals(fn=plot_fn)
			if fn == plot_fn:
				alpha=1
				ls = "-"
				label = f"Efficient Frontier of {fn.basebasename}"
			else:
				alpha=0.7
				ls = "--"
				label = f"Values of {plot_fn.basebasename}\non Efficient Frontier of {fn.basebasename}"

			ax.plot(
			        x_raw,y_raw,
			        label=label,
			        color=color_map[fn],
			        marker=marker_map[fn],
			        alpha=alpha,
			        ls=ls
			        )

		line_artists.extend(ax.collections.copy() + ax.lines.copy())
		legend_dict = {artist.properties().get('label') : artist for artist in line_artists}
		plt.legend(legend_dict.values(),legend_dict.keys(),loc='best')
		ax.set_xlabel(f"{plot_fn.basebasename}-Utility for type-{theta_low}")
		ax.set_ylabel(f"{plot_fn.basebasename}-Utility\nfor Entire Population")
		ax.set_title(f"'{plot_fn.basebasename}' Values of Other Efficient Solutions")

	plt.show(block=False)
	if saving:
		plt.savefig(f"figures/different_frontiers_{generate_file_label()}.pdf", bbox_inches='tight')



best,frontiers,dist = solve_multiple_frontier()
num_in_each_frontier = {fn.__name__ : len(frontier.instances) for fn, frontier in frontiers.items()}
while min(num_in_each_frontier.values()) < 7:
	print("REPEATING SOLUTION TO GET MORE INTERESTING FRONTIERS")
	best,frontiers,dist = solve_multiple_frontier()
	num_in_each_frontier = {fn.__name__ : len(frontier.instances) for fn, frontier in frontiers.items()}
# pickle.dump(dist,open('last_dist.pickle','wb'))

print(num_in_each_frontier)
# print([len(front.instances) for front in frontiers[SWB_dist]])
# frontier = frontiers[SWB_dist][np.argmax([len(front.instances) for front in frontiers[SWB_dist]])]
# fig,ax,xx,yy = plot_frontier(frontiers[SWB_dist])

# plot_frontiers(best,frontiers)

saving = True
plot_tradeoffs(best,frontiers,saving=saving)


# frontier = frontiers[SWB_dist]
# fig,ax = plot_frontier(frontier)


