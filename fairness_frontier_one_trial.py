import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.legend_handler import HandlerTuple
from datetime import datetime
import pickle
import sys
from subset_helper import bax
from function_factories_frontier import sum_log_prob_weighted_factory, sum_prob_weighted_factory,sum_distance_weighted_factory,mean_distance_function,mean_probability_function,type_covariance_function, price_of_fairness_distance_function, price_of_fairness_probability_function,markers,covariance_factory, linear_combination_of_objectives
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True


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
sizeRegion = 1
theta_low = 0
theta_high = 1

dumpFileName='last_dist_frontier_one_trial.pickle'
saving = True
loading = True
dumping = not loading


# ### Covariance Linear Combination (start)
# title = f"Efficient Frontier of $\\sum P(Success)$\nwith Objectives Minimizing $(1-\\gamma)P(Success) + \\gamma Cov(type,P(Success))$"
# colorbar_label= "$\\gamma$"
# trial_label = f"Cov_efficiency"
# nIndiv = 50
# nFac = 20
# nSelectedFac = 4
# theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
# nIndiv = len(theta)
# theta_low_indices = [i for i in range(nIndiv) if theta[i]==theta_low]
# beta = [0,1,-10] #[beta0, beta_theta >0, beta_r <0]
# weights = [0,.9,0.95,.98] #should all be in [0,1]
# log_prob_obj_fns = [sum_log_prob_weighted_factory(w) for w in weights]
# prob_obj_fns = [sum_prob_weighted_factory(w) for w in weights]
# dist_obj_fns= [sum_distance_weighted_factory(w) for w in weights]
# SWB_dist = dist_obj_fns[0]
# SWB_prob = prob_obj_fns[0]
# SWB_logprob = log_prob_obj_fns[0]
# SWB_functions = [SWB_prob]#[SWB_dist, SWB_prob,SWB_logprob]
# cov_fn = covariance_factory()
# prob_plus_lambda_cov = [linear_combination_of_objectives(SWB_prob,cov_fn,w) for w in weights]
# objective_functions = prob_plus_lambda_cov
# all_objectives_for_frontier_and_obj = list(set(SWB_functions + objective_functions))
# ### Covariance Linear Combination (end)

### SWN (start)
title = f"Efficient Frontier of $\\sum P(Success)$\nwith Objectives Minimizing $\\alpha$-Fair $\\sum\\log(P(Success))$"
colorbar_label= "$\\alpha$"
trial_label = f"log_prob_alpha_fair"
nIndiv = 50
nFac = 20
nSelectedFac = 4
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
nIndiv = len(theta)
theta_low_indices = [i for i in range(nIndiv) if theta[i]==theta_low]
beta = [0,1,-10] #[beta0, beta_theta >0, beta_r <0]
weights = [0,0.5,0.75,0.9] #should all be in [0,1]
log_prob_obj_fns = [sum_log_prob_weighted_factory(w) for w in weights]
prob_obj_fns = [sum_prob_weighted_factory(w) for w in weights]
dist_obj_fns= [sum_distance_weighted_factory(w) for w in weights]
SWB_dist = dist_obj_fns[0]
SWB_prob = prob_obj_fns[0]
SWB_logprob = log_prob_obj_fns[0]
SWB_functions = [SWB_prob]#[SWB_dist, SWB_prob,SWB_logprob]
cov_fn = covariance_factory()
prob_plus_lambda_cov = [linear_combination_of_objectives(SWB_prob,cov_fn,w) for w in weights]
objective_functions = log_prob_obj_fns
all_objectives_for_frontier_and_obj = list(set(SWB_functions + objective_functions))
### SWN (end)





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
			self.uvals = {fn: fn(self) for fn in all_objectives_for_frontier_and_obj}
		if fstars is not None:
			self.fstars = fstars
		else:
			self.fstars = {fn: sum(self.uvals[fn]) for fn in all_objectives_for_frontier_and_obj}
		if uvals_type0 is not None:
			self.uvals_type0 = uvals_type0
		else:
			self.uvals_type0 = {fn: self.uvals[fn][theta_low_indices] for fn in all_objectives_for_frontier_and_obj}
		if fstars_type0 is not None:
			self.fstars_type0 = fstars_type0
		else:
			self.fstars_type0 = {fn: sum(self.uvals_type0[fn]) for fn in all_objectives_for_frontier_and_obj}
	def combine(self, other_instance):
		if other_instance is not None:
			# print(f"len(self.rvals: {len(self.rvals)}, len(other_instance.rvals): {len(other_instance.rvals)}")
			return Instance(
							rvals = self.rvals + other_instance.rvals,
							yvals = self.yvals + other_instance.yvals,
							uvals = add_dict_lists(self.uvals,other_instance.uvals),
							fstars = {fn: sum(self.uvals[fn]) for fn in all_objectives_for_frontier_and_obj},
							uvals_type0 = add_dict_lists(self.uvals_type0,other_instance.uvals_type0),
							fstars_type0 = {fn: sum(self.uvals_type0[fn]) for fn in all_objectives_for_frontier_and_obj}
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
		return {fn: (self.fstars[fn] > other_instance.fstars[fn] and self.fstars_type0[fn] > other_instance.fstars_type0[fn]) for fn in all_objectives_for_frontier_and_obj}
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

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_tradeoff(best,frontiers,saving=False):
	# fig, ax = plt.subplots(figsize=(10,10))
	fig = plt.figure(figsize=(10,10))
	axes = {}

	cmap = plt.cm.get_cmap('cividis')
	# norm = Normalize(vmin=min([weight for weight in weights if weight > 0]), vmax=max(weights))
	norm = MidpointNormalize(vmin=min(weights),vmax=max(weights),midpoint=np.median(weights))
	sm = ScalarMappable(norm=norm, cmap=cmap)
	# sm.to_rgba(other_fn.weight)


	# colors = get_n_colors(len(all_objectives_for_frontier_and_obj))
	color_map = {fn.weightstr : sm.to_rgba(fn.weight) for fn in best} #colors.pop()
	# marker_map = {fn : f"${fn.marker}^{{{fn.weight}}}$" for fn in best}
	line_artists = []
	# point_legend_helper = {fn : (color_map[fn],fn.marker,fn.basename) for fn in best}
	scat = {fn : [] for fn in frontiers}

	scat_xdata = [instance.fstars_type0[fn] for i,(fn,frontier) in enumerate(frontiers.items()) for (other_fn,instance) in best.items()]
	scat_ydata = [instance.fstars[fn] for i,(fn,frontier) in enumerate(frontiers.items()) for (other_fn,instance) in best.items()]
	marker_size = 0.05
	def get_nonoverlapping_positions(x_data, y_data):
		new_x = x_data.copy()
		new_y = y_data.copy()
		numel = len(new_x)
		for index, (x, y) in enumerate(zip(new_x,y_data)):
			to_change = []
			for index2 in range(index+1,numel):
				other_x = new_x[index2]
				other_y = y_data[index2]
				if abs(x-other_x) < marker_size and abs(y-other_y) < marker_size:
					to_change.append(index2)
			for index2 in to_change:
				# print(f"Adjusting new_x[{index2}] = {new_x[index2]},y_data[{index2}] = {y_data[index2]}")
				new_x[index2] += marker_size*0.2
				new_y[index2] += marker_size*0.05
		return new_x,new_y
	scat_xdata_adjusted,scat_ydata_adjusted = get_nonoverlapping_positions(scat_xdata,scat_ydata)
	ctr = 0
	for i, (fn, frontier) in enumerate(frontiers.items()):
		ax = fig.add_subplot(111,label=fn.__name__,frame_on=False)
		cbar = fig.colorbar(sm,ax=ax,drawedges=False)
		cbar.ax.set_ylabel(f"{colorbar_label}")
		axes[fn] = ax
		x_raw = frontier.fstarvals_type0()
		y_raw = frontier.fstarvals()
		print(f"Plotting data for {fn.__name__}")
		color = color_map[fn.weightstr]
		linename = f"{fn.__name__} Efficient Frontier"
		ax.plot(x_raw,y_raw,label=linename,color="black",marker="4",alpha=0.7)
		line_artists.extend(ax.collections.copy() + ax.lines.copy())
		print(f"Scattering for {fn.__name__}")
		for other_fn, instance in best.items():
			if True: #other_fn != fn:
				print(f"{other_fn.__name__}, ({instance.fstars_type0[fn]},{instance.fstars[fn]}), marker: {other_fn.weighted_marker}")
				x_inst = instance.fstars_type0[fn]# + np.random.rand()/200 #jittering
				# x_inst_adjusted = x_inst
				y_inst = instance.fstars[fn]# + np.random.rand()/200 #jittering
				# y_inst_adjusted = scat_ydata_adjusted[ctr]
				# x_inst = scat_xdata_adjusted[ctr]
				# y_inst = scat_ydata_adjusted[ctr]
				ctr += 1
				# label = other_fn.__name__
				label = f"{other_fn.__name__}"
				scatcolor = color_map[other_fn.weightstr]
				scatplot = ax.scatter([x_inst],[y_inst],label=label,s=20**2,color=scatcolor,marker=other_fn.weighted_marker,alpha=0.8)
				# arrow_size = marker_size/100
				# arrow = ax.arrow(x_inst, y_inst_adjusted,0,y_inst-y_inst_adjusted, color='red',alpha=0.3, width=arrow_size*0.1, head_width=arrow_size, head_length=marker_size*0.5, zorder=0,length_includes_head=True)

				fakelabel = f"{other_fn.basename}"
				fakescatplot = matplotlib.lines.Line2D([], [],label=fakelabel,markersize=20,color=scatcolor,marker=other_fn.basemarker,alpha=0.8,linestyle='None')
				# scat[fn].append(scatplot)
				scat[fn].append(fakescatplot)
# fakescatplot = matplotlib.lines.Line2D([], [],label="hi",markersize=20,color="red",marker="+",alpha=0.8,linestyle='None')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.tick_params(axis='x', labelcolor=color)
		ax.tick_params(axis='y', labelcolor=color)
	# artists = [artist for ax in axes for artist in ax.collections.copy() + ax.lines.copy()]
	legend_dict = {artist.properties().get('label') : artist for artist in line_artists}
	scat_dict = {}
	for fn,art_list in scat.items():
		for artist in art_list:
			artist_lab = artist.properties().get('label')
			scat_dict_key = f"Optimal {artist_lab}"
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
	plt.title(title)
	plt.show(block=False)
	if saving:
		plt.savefig(f"figures/frontier_{trial_label}_{generate_file_label()}.pdf", bbox_inches='tight')
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


