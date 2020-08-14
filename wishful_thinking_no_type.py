from casadi import Opti,log,exp
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from subset_helper import bax
from function_helpers import setmeta
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True


sizeRegion = 1

# nIndiv = 100
# nFac = 10
theta_low = 0
theta_high = 1


# sum prob
# nSelectedFac = 1
# trial_label = "Logistic Pitfall"
# indiv = np.array([[0.2,0.2],[0.2,0.8]])
# fac = np.array([[0.2,0.1],[0.2,0.9]])
# theta = [theta_high,theta_low]
# nIndiv = indiv.shape[0]
# nFac = fac.shape[0]
# beta = [0,3,-10] #[beta0, beta_theta >0, beta_r <0]
# facLabels = ["A","B"]


nSelectedFac = 2
trial_label = "Maximum $\\sum \\log(P(Success))$"
indiv1 = np.array([[0.5,0.025]])
subNIndiv = 8
rad = 0.05
dx = 0.4
dy = .65
centers = [[0.5-dx/2,dy],[0.5+dx/2,dy]]
extra_displacements = np.linspace(0,2.5*rad,subNIndiv)
# extra_displacements = 2*rad*np.random.rand(subNIndiv)
indiv2 = np.array([[center[0] + (rad + extra_displacements[i])*np.cos(2*np.pi*(0.25+(i/subNIndiv))), center[1]+ (rad + extra_displacements[i])*np.sin(2*np.pi*(0.25 + (i/subNIndiv)))] for center in centers for i in range(subNIndiv)])
indiv = np.concatenate((indiv1,indiv2))
fac = np.concatenate((np.array([[0.5,0.1]]),centers.copy() ))
nIndiv = indiv.shape[0]
nFac = fac.shape[0]
theta = theta_low
beta = [4,0,-18] #[beta0, beta_theta >0, beta_r <0]
facLabels = ["A","B","C"]













@setmeta(extra_info="Maximize Probability of Success of All Individuals",name="$\\sum \\log(P(Success))$",indiv_name="$\\log(P(Success))$")
def logPredProb(rr):
	u = [-log(1+exp(-beta[0] - beta[1]*theta - beta[2]*rr[i])) for i in range(len(rr))]
	return u


@setmeta(extra_info="Maximize $\\sum_{individuals} P(success)$",name="$\\sum P(Success)$",indiv_name="$P(Success)$")
def predProb(rr):
	u = [1/(1+exp(-beta[0] - beta[1]*theta - beta[2]*rr[i])) for i in range(len(rr))]
	return u


obj_fns = [logPredProb,predProb]




dist = np.array([[ (indiv[i,0] - fac[j,0])**2 + (indiv[i,1] - fac[j,1])**2 for j in range(0,nFac)] for i in range(0,nIndiv)]) # nIndiv X nFac


def solve_enumerative(obj_fn):
	bbax=bax()
	yvals = np.zeros((nFac),dtype=int)
	rvals = np.zeros((nIndiv))
	uvals = np.full((nIndiv),-np.inf)
	for ind, gp in enumerate(bbax.bax_gen(nFac,nSelectedFac)):
		gp = gp.toList()
		xlist = [ gp[np.argmin([dist[i][j] for j in gp])] for i in range(nIndiv)]
		r = [dist[i][xlist[i]] for i in range(nIndiv)]
		u = obj_fn(r)
		print(f"Obj is {sum(u)}")
		if sum(u) > sum(uvals):
			uvals = u
			rvals = r
			yvals = np.zeros((nFac),dtype=int)
			yvals[gp] = 1
	print(f"Best obj is {sum(uvals)}")
	fstar = sum(uvals)
	prob_success = [1/(1+exp(-beta[0] - beta[1]*theta - beta[2]*rvals[i])) for i in range(nIndiv)]
	print(f"Solved enumeratively")
	return yvals,rvals,uvals,prob_success,fstar, obj_fn.extra_info

def get_n_colors(number_desired_colors):
	if number_desired_colors < 10:
		default = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		colors = default[0:number_desired_colors]
	else:
		# https://matplotlib.org/tutorials/colors/colormaps.ht
		cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
		colors = [cmap(i) for i in range(number_desired_colors)]
	return colors

def plot_region_and_curves(the_fn,saving=False,extra_info=""):
	extra_info = solns[the_fn]['extra_info']
	yvals = solns[the_fn]['yvals']
	prob_success = solns[the_fn]['prob_success']
	colors = get_n_colors(6)
	# color_type_map = {theta : colors.pop()}
	markers = ['|','3','$/$','$\\backslash$','|','+','x','P','*','+','x','_']
	marker_sizes = [80,45,60,60]
	# markers = ['\\times','\\bigcirc','+','\\Box','\\dagger','\\Diamond','\\oplus','\\wedge','\\oslash','\\bullet','\\odot','\\triangleright','\\otimes',]
	individual_color = colors.pop()
	facility_color_map = {yval : colors.pop() for yval in set(yvals)}
	function_color_map = {fn : colors.pop() for fn in obj_fns}
	function_marker_map = {fn : markers.pop(0) for fn in obj_fns}
	function_markersize_map = {fn : marker_sizes.pop(0) for fn in obj_fns}
	facility_size_options = [5**2,9**2]
	facility_states = ["Omitted", "Selected"]

	nPlots = 1 + len(obj_fns)
	gs = gridspec.GridSpec(nrows=nPlots,ncols=1,height_ratios=[0.75] + [1 for i in range(nPlots-1)],hspace=0.01)
	fig = plt.figure(figsize=(5.5,5*nPlots))
	axes = [fig.add_subplot(gs[i]) for i in range(nPlots)]

	# fig.subplots_adjust(hspace=0.3)

	## Plot Region
	def plotRegion():
		ax = axes[0]
		ax.set_title(f"Optimizing {the_fn.name}",fontsize=11)
		ax.set_xlim(0,1)
		ax.set_xticks([])
		ax.set_yticks([])
	plotRegion()

	## Plot Individuals
	labelIndividuals = "Individual"
	def plotIndividuals():
		ax = axes[0]
		# individual_colors = [color_type_map[theta] for i in range(nIndiv)]
		individual_labels = [f"{labelIndividuals} (type {theta})" for coords in indiv]
		for i,coords in enumerate(indiv):
			ax.scatter(*coords,marker = "*",c=individual_color,label=individual_labels[i])
	plotIndividuals()

	## Plot Facilities
	labelFacility = "Facility"
	def plotFacilities():
		ax = axes[0]
		facility_sizes = [facility_size_options[yval] for yval in yvals]
		facility_colors = [facility_color_map[yval] for yval in yvals]
		facility_labels = [f"{facility_states[yval]} {labelFacility}" for yval in yvals]
		for i, coords in enumerate(fac):
			ax.scatter(*coords,marker = "+",c=[facility_colors[i]],s=[facility_sizes[i]],label=facility_labels[i])
		for i,lab in enumerate(facLabels):
			ax.text(fac[i][0]+.01,fac[i][1]-.01,lab,fontsize=10)
	plotFacilities()

	## Plot Objectives
	def plotObjectives():
		ax = axes[0]
		obj_scatter = ax.scatter(indiv[:,0],indiv[:,1],marker="o",s=12**2,lw=2,alpha=0.5,c=prob_success,cmap="cividis",label="no_legend")
		obj_scatter.set_facecolor('none')
		cbar = fig.colorbar(obj_scatter,ax=ax)
		cbar.set_clim(0,1)
		cbar.ax.set_ylabel('P(success)')
	plotObjectives()

	def plotRegionLegend():
		ax = axes[0]
		legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy() if "no_legend" not in artist.properties().get('label')}
		ax.legend(legend_dict.values(),legend_dict.keys(),loc='best',fontsize='x-small')
	plotRegionLegend()
	## Plot Objectives
	def plotCurves():
		for plot_ind,plot_obj_fn in enumerate(obj_fns):
			ax = axes[plot_ind+1]
			rspace = np.linspace(-0.005,0.6,num=1000)
			uspace = plot_obj_fn(rspace)
			ax.plot(rspace,uspace,color='black',label=f"{plot_obj_fn.indiv_name}",zorder=0,alpha=1,lw=0.5)
			best_val = max([sum(plot_obj_fn(solns[fn]['rvals'])) for fn in solns])
			for fn,soln in solns.items():
				r = solns[fn]['rvals']
				u = plot_obj_fn(r)
				sumUtilities = sum(u)
				optimality = "suboptimal" if sumUtilities < best_val else "optimal"
				ax.scatter(r,u,color=function_color_map[fn],marker=function_marker_map[fn],label=f"Individual {plot_obj_fn.indiv_name} from Facilities {', '.join(soln['chosen_fac'])}\n{plot_obj_fn.name}={sumUtilities:.2f} ({optimality})",zorder=1,alpha=0.65,edgecolors=None,s=function_markersize_map[fn])


			legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy() if "no_legend" not in artist.properties().get('label')}
			ax.legend(legend_dict.values(),legend_dict.keys(),loc='best',fontsize='x-small')
			if plot_ind == len(obj_fns)-1:
				ax.set_xlabel("distance from nearest selected facility")
			else:
				ax.set_xticks([])
			ax.set_ylabel(f"{plot_obj_fn.indiv_name}")
			# ax.set_title(f"\n\n\n{plot_obj_fn.name} vs Distance")
	plotCurves()
	
	# fig.tight_layout()
	plt.show(block=False)
	if saving:
		plt.savefig(f"figures/anti_triage_{generate_file_label()}.pdf", bbox_inches='tight')


	return fig,axes


def generate_file_label():
	# weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_stamp = f"at_{timestamp}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nSelectedFac}of{nFac}"
	return trial_stamp





saving = True
solns = {}
for obj_fn in obj_fns:
	solns[obj_fn] = {}
	solns[obj_fn]['yvals'],solns[obj_fn]['rvals'],solns[obj_fn]['uvals'],solns[obj_fn]['prob_success'],solns[obj_fn]['fstar'],solns[obj_fn]['extra_info']=solve_enumerative(obj_fn)
	solns[obj_fn]['selected'] = [i for i in range(len(solns[obj_fn]['yvals'])) if solns[obj_fn]['yvals'][i]==1]
	solns[obj_fn]['chosen_fac'] = [facLabels[i] for i in solns[obj_fn]['selected']]
fig,axes = plot_region_and_curves(the_fn=logPredProb,saving=saving)


# fig,ax = plot_region("(enumerative method)")


