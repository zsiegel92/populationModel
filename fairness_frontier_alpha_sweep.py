import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.legend_handler import HandlerTuple
from datetime import datetime
import pickle
from subset_helper import bax
from function_factories_alpha import sum_log_prob_weighted_factory, sum_prob_weighted_factory,sum_distance_weighted_factory,covariance_factory, linear_combination_of_objectives
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
dumpFileName='last_dist_frontier_one_trial_10pts.pickle'
saving = True
loading = True
dumping = not loading


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
weights = np.linspace(0,1,2000) #should all be >= 1
weightstr = "linspace(0,1,2000)"
# weightstr = str(weights)
log_prob_obj_fns = [sum_log_prob_weighted_factory(w) for w in weights]
prob_obj_fns = [sum_prob_weighted_factory(w) for w in weights]
dist_obj_fns= [sum_distance_weighted_factory(w) for w in weights]
SWB_dist = dist_obj_fns[0]
SWB_prob = prob_obj_fns[0]
SWB_logprob = log_prob_obj_fns[0]
SWB_functions = [SWB_prob]#[SWB_dist, SWB_prob,SWB_logprob]
cov_fn = covariance_factory()
prob_plus_lambda_cov = [linear_combination_of_objectives(SWB_prob,cov_fn,w) for w in weights]

objective_functions = log_prob_obj_fns + prob_obj_fns + dist_obj_fns + prob_plus_lambda_cov
objective_functions = prob_obj_fns

# indices = [
# 	{'name':'Mean Distance','key': 'mean_dist', 'shared':False,'fn':mean_distance_function},
# 	{'name':'Mean Success Prob','key': 'mean_prob','shared':False,'fn':mean_probability_function},
# 	{'name':'Covariance(type,success)','key': 'cov','shared':True,'fn':type_covariance_function},{'name':'Price of Fairness: Distance','key': 'pof_dist','shared':True,'fn':price_of_fairness_distance_function},
# 	{'name':'Price of Fairness: Probability','key': 'pof_prob','shared':True,'fn':price_of_fairness_probability_function}
# 	]


objective_function_names = {fn.__name__ : fn for fn in objective_functions}

# fairness_ratings = [10**10 + weight for weight in weights]+ [10**5 + weight for weight in weights] + weights #arbitrary, but roughly increasing with fairness-inducing tendency
# fairness_strengths = dict(zip(objective_functions,fairness_ratings)) # to create a consistent order in plot legends

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
	def fstarvals(self):
		return [instance.fstars[self.fn] for instance in self.instances]
	def fstarvals_type0(self):
		return [instance.fstars_type0[self.fn] for instance in self.instances]
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
	if loading:
		dist = pickle.load(open(dumpFileName,'rb'))
	else:
		dist = np.array([[person.dist(facility) for facility in fac] for person in indiv])
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
	# weights_label = "_".join(map(str,weights))
	weights_label = weightstr
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



def plot_tradeoff(best,frontiers,saving=False):
	# fig, ax = plt.subplots(figsize=(10,10))
	fig = plt.figure(figsize=(10,10))
	axes = {}
	colors = get_n_colors(len(best))
	color_map = {fn.weightstr : colors.pop() for fn in best}
	# marker_map = {fn : f"${fn.marker}^{{{fn.weight}}}$" for fn in best}
	line_artists = []
	# point_legend_helper = {fn : (color_map[fn],fn.marker,fn.basename) for fn in best}
	scat = {fn : [] for fn in best}
	for i, (fn, frontier) in enumerate(frontiers.items()):
		ax = fig.add_subplot(111,label=fn.__name__,frame_on=False)
		axes[fn] = ax
		line_color = color_map[fn.weightstr]
		def plot_frontier():
			x_raw = frontier.fstarvals_type0()
			y_raw = frontier.fstarvals()
			print(f"Plotting data for {fn.__name__}")
			linename = f"$\\sum$ {fn.basebasename} Efficient Frontier"
			ax.plot(x_raw,y_raw,label=linename,color=line_color,marker="4",alpha=0.7,zorder=1)
		plot_frontier()
		line_artists.extend(ax.collections.copy() + ax.lines.copy())
		print(f"Scattering for {fn.__name__}")
		cmap = plt.cm.get_cmap('cividis')
		norm = Normalize(vmin=min(weights), vmax=max(weights))
		sm = ScalarMappable(norm=norm, cmap=cmap)
		def scatter_objectives():
			scat_x_vals = [instance.fstars_type0[fn] for other_fn,instance in best.items()]
			scat_y_vals = [instance.fstars[fn] for other_fn,instance in best.items()]
			scat_markers = [other_fn.mpl_marker for other_fn,instance in best.items()]
			scat_labels = [f"{other_fn.basename}" for other_fn,instance in best.items()]
			scat_colors = [sm.to_rgba(other_fn.weight) for other_fn, instance in best.items()]
			# scat_sizes = [(1- other_fn.weight) * 8**2  for other_fn, instance in best.items()]
			ax.scatter(scat_x_vals,scat_y_vals, label = scat_labels[0], marker=scat_markers[0],s=8**2,alpha=1,c=scat_colors,zorder=2)
		scatter_objectives()
		def plot_alpha_labels():
			instances = tuple(set(best.values()))
			labels_at_inst = {inst: f"${min(ll:=[the_fn.weight for the_fn,the_instance in best.items() if the_instance == inst]):.02f}-{max(ll):.02f}$" for inst in instances}
			inst_x_values = {inst : inst.fstars_type0[fn] for inst in instances}
			inst_y_values = {inst : inst.fstars[fn] for inst in instances}
			for inst in instances:
				xoffset= 0
				yoffset=0.01
				ax.text(inst_x_values[inst] + xoffset,inst_y_values[inst] + yoffset,labels_at_inst[inst],fontsize=8,zorder=10)
		plot_alpha_labels()
		scat[fn] = [matplotlib.lines.Line2D([], [],markersize=10,color=sm.to_rgba(other_fn.weight), marker=other_fn.mpl_marker,alpha=0.8,linestyle='None') for i,(other_fn,instance) in enumerate(best.items())]
		cbar = fig.colorbar(sm,ax=ax,drawedges=False)
		# cbar.set_clim(0,1)
		cbar.ax.set_ylabel('$\\alpha$')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.tick_params(axis='x', labelcolor=line_color)
		ax.tick_params(axis='y', labelcolor=line_color)
	# artists = [artist for ax in axes for artist in ax.collections.copy() + ax.lines.copy()]
	legend_dict = {artist.properties().get('label') : artist for artist in line_artists}
	scat_dict = {}
	for fn,art_list in scat.items():
		for artist in art_list:
			artist_lab = artist.properties().get('label')
			scat_dict_key = f"Efficient $\\alpha$-Fair Solutions"
			if not scat_dict.get(scat_dict_key):
				scat_dict[scat_dict_key] = []
			scat_dict[scat_dict_key].append(artist)
	for k,l in scat_dict.items():
		scat_dict[k].sort(key = lambda artist: str(artist.get_markeredgecolor()))
		scat_dict[k] = tuple(l)
	legend_dict = {**legend_dict,**scat_dict}
	print(scat_dict)
	plt.legend(legend_dict.values(),legend_dict.keys(),loc='lower left',handler_map={tuple: HandlerTuple(ndivide=None)},labelspacing=0.9,handlelength=3.5) #,handletextpad=1.0,loc="upper right"bbox_to_anchor=(0.5, 0), ncol=len(legend_dict),fontsize='small'
	# plt.setp(fig.get_axes(), xticks=[], yticks=[])
	ax.set_xlabel(f"Utility to type-$0$ Individuals")
	ax.set_ylabel(f"Utility to Population")
	weight_string = ", ".join([str(weight) for weight in weights])
	title = f"Efficient Frontier and $\\alpha$-Fair Efficient Solutions" #+f"\n(weights: {weight_string})"
	plt.title(title)
	plt.show(block=False)
	if saving:
		plt.savefig(f"figures/alpha_sweep_frontier_{generate_file_label()}.pdf", bbox_inches='tight')
	return fig,axes,scat



best,frontiers,dist = solve_multiple_frontier()
num_in_each_frontier = {fn.__name__ : len(frontier.instances) for fn, frontier in frontiers.items()}
interesting_threshold = 9
while min(num_in_each_frontier.values()) < interesting_threshold:
	print(f"REPEATING SOLUTION TO GET MORE INTERESTING FRONTIERS (got {num_in_each_frontier.values()}, want >={interesting_threshold})")
	best,frontiers,dist = solve_multiple_frontier()
	num_in_each_frontier = {fn.__name__ : len(frontier.instances) for fn, frontier in frontiers.items()}

if dumping:
	pickle.dump(dist,open(dumpFileName,'wb'))

print(num_in_each_frontier)
# print([len(front.instances) for front in frontiers[SWB_dist]])
# frontier = frontiers[SWB_dist][np.argmax([len(front.instances) for front in frontiers[SWB_dist]])]
# fig,ax,xx,yy = plot_frontier(frontiers[SWB_dist])

# plot_frontiers(best,frontiers)


fig,axes,scat = plot_tradeoff(best,frontiers,saving=saving)

ax = axes[SWB_prob]
pts = scat[SWB_prob]
# frontier = frontiers[SWB_dist]
# fig,ax = plot_frontier(frontier)

# plt.savefig(f"figures/alpha_sweep_frontier_{generate_file_label()}.pdf", bbox_inches='tight')
