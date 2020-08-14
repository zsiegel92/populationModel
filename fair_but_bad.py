from casadi import Opti,log,exp
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from subset_helper import bax
from function_helpers import setmeta
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True


sizeRegion = 1

# nIndiv = 100
# nFac = 10
theta_low = 0
theta_high = 1


### Sum Logprob
nSelectedFac = 2
trial_label = "Maximum $\\sum \\log(P(Success))$"
indiv1 = np.array([[0.5,0.05]])
subNIndiv = 10
rad = 0.05
dx = 0.4
dy = .75
centers = [[0.5-dx/2,dy],[0.5+dx/2,dy]]
indiv2 = np.array([[center[0] + rad*np.cos(2*np.pi*(0.25+(i/subNIndiv))), center[1]+ rad*np.sin(2*np.pi*(0.25 + (i/subNIndiv)))] for center in centers for i in range(subNIndiv)])
indiv = np.concatenate((indiv1,indiv2))
fac = np.concatenate((np.array([[0.5,0.1]]),centers.copy() ))
nIndiv = indiv.shape[0]
nFac = fac.shape[0]
theta = [theta_low]*nIndiv
beta = [4,0,-18] #[beta0, beta_theta >0, beta_r <0]
facLabels = ["A","B","C"]


@setmeta(extra_info="$\\sum \\log(P(success))$")
def sum_logprob(rr):
	u = [-log(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nIndiv)]
	return u
obj_fn=sum_logprob


# Simplest Min Cov

# nSelectedFac = 1
# trial_label = "Minimum Cov(Type,P(Success))"
# indiv = np.array([[0.4,0.5],[0.8,0.5]])
# fac = np.array([[0.2,0.5],[0.6,0.5]])
# theta = [theta_low,theta_high]
# nIndiv = indiv.shape[0]
# nFac = fac.shape[0]
# beta = [0,1,-10] #[beta0, beta_theta >0, beta_r <0]
# facLabels = ["A","B"]

## More Complex Min Cov
# nSelectedFac = 1 #1 or 2
# trial_label = "Minimum Cov(Type,P(Success))"
# indiv = np.array([[0.2,0.5],[0.3,0.4],[0.3,0.6],[0.8,0.4],[0.8,0.6],[0.9,0.5]])
# fac = np.array([[0.1,0.5],[0.3,0.5],[0.8,0.5]])
# theta = [theta_low]*3 +[theta_high]*3
# nIndiv = indiv.shape[0]
# nFac = fac.shape[0]
# beta = [0,1,-10] #[beta0, beta_theta >0, beta_r <0]
# facLabels = ["A","B","C"]

### Min Cov
# @setmeta(extra_info="Minimum Cov(type,P(Success))")
# def neg_covariance(rr):
# 	avg_theta = (1/nIndiv) * sum(theta) #should be 0.5
# 	u =np.array( [-(1/nIndiv)* (theta[i] - avg_theta) / (1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nIndiv)])
# 	return u
# obj_fn = neg_covariance


# Min GE
# nSelectedFac = 1
# trial_label = "Minimum Generalized Entropy"
# indiv = np.array([[0.5,0.1],[0.8,0.8],[0.75,0.9],[0.2,0.8],[0.25,0.9]])
# indiv1 = np.array([[0.5,0.1]])
# subNIndiv = 5
# rad = 0.1
# centers = [[0.25,0.75],[0.75,0.75]]
# indiv2 = np.array([[center[0] + rad*np.cos(2*np.pi*(0.25+(i/subNIndiv))), center[1]+ rad*np.sin(2*np.pi*(0.25 + (i/subNIndiv)))] for center in centers for i in range(subNIndiv)])
# indiv = np.concatenate((indiv1,indiv2))
# fac = np.concatenate((np.array([[0.5,0.15]]),centers.copy() ))
# nIndiv = indiv.shape[0]
# nFac = fac.shape[0]
# theta = [theta_low]*nIndiv
# beta = [0,5,-1] #[beta0, beta_theta >0, beta_r <0]
# facLabels = ["A","B","C"]

# @setmeta(extra_info="Minimum Generalized Entropy")
# def GE_fn(rr):
# 	alpha = 0.01
# 	u =np.array( [1 / (1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nIndiv)])
# 	mu = np.mean(u)
# 	u = (-1/(nIndiv*alpha *(alpha-1))) * np.array([(ui/mu)**alpha - 1 for ui in u])
# 	return u
# obj_fn = GE_fn


dist = np.array([[ (indiv[i,0] - fac[j,0])**2 + (indiv[i,1] - fac[j,1])**2 for j in range(0,nFac)] for i in range(0,nIndiv)]) # nIndiv X nFac




def solve_enumerative(obj_fn):
	bbax=bax()
	extra_info = ""
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
	prob_success = [1/(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*rvals[i])) for i in range(nIndiv)]
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

def plot_region(title_extra="",saving=False,extra_info=""):
	colors = get_n_colors(6)
	fig, ax = plt.subplots(figsize=(10,10))
	plt.axis('off')
	title = f"Bad Decisions: Facilities and Individuals"
	if len(title_extra) > 0:
		title = f"{title}\n{title_extra}"
	plt.title(title)

	## Plot Region
	# minx = min(xx for (xx,yy) in indiv)
	# maxx = max(xx for (xx,yy) in indiv)
	# miny = min(yy for (xx,yy) in indiv)
	# maxy = max(yy for (xx,yy) in indiv)
	# sizex = maxx-minx
	# sizey = maxy-miny
	# margin = .01
	# region = patches.Rectangle((minx-margin*sizex,miny-margin*sizey),(1+margin)*sizex,(1+margin)*sizey,linewidth=1,edgecolor=colors.pop(),facecolor="none")
	region = patches.Rectangle((0,0),sizeRegion,sizeRegion,linewidth=1,edgecolor=colors.pop(),facecolor="none")
	ax.add_patch(region)

	## Plot Objectives
	def plotObjectives():
		obj_scatter = plt.scatter(indiv[:,0],indiv[:,1],marker="o",s=14**2,lw=2,alpha=0.5,c=prob_success,cmap="cividis",label="no_legend")
		obj_scatter.set_facecolor('none')
		cbar = plt.colorbar(obj_scatter)
		# plt.clim(0,1)
		cbar.ax.set_ylabel('P(success)')
	plotObjectives()

	## Plot Individuals
	labelIndividuals = "Individual"
	def plotIndividuals():
		color_weight_map = {ttheta : colors.pop() for ttheta in set(theta)}
		individual_colors = [color_weight_map[ttheta] for ttheta in theta]
		individual_labels = [f"{labelIndividuals} type {ttheta}" for coords,ttheta in zip(indiv,theta)]
		for i,coords in enumerate(indiv):
			plt.scatter(*coords,marker = "*",c=[individual_colors[i]],label=individual_labels[i])
	plotIndividuals()

	## Plot Facilities
	labelFacility = "Facility"
	def plotFacilities():
		size_options = [5**2,9**2]
		facility_sizes = [size_options[yval] for yval in yvals]
		facility_color_map = {yval : colors.pop() for yval in set(yvals)}
		facility_colors = [facility_color_map[yval] for yval in yvals]
		facility_states = ["Omitted", "Selected"]
		facility_labels = [f"{facility_states[yval]} {labelFacility}" for yval in yvals]
		for i, coords in enumerate(fac):
			plt.scatter(*coords,marker = "+",c=[facility_colors[i]],s=[facility_sizes[i]],label=facility_labels[i])
		for i,lab in enumerate(facLabels):
			plt.text(fac[i][0],fac[i][1]+.01,s=lab)
	plotFacilities()

	## Create Legend
	# legend_dict = {legendTitle : artist for artist in ax.collections.copy() + ax.lines.copy() if "no_legend" not in (legendTitle:=artist.properties().get('label'))} #Python 3.8
	legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy() if "no_legend" not in artist.properties().get('label')} #Python 3.7
	plt.legend(legend_dict.values(),legend_dict.keys(),loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(legend_dict),fontsize='small') #,loc="upper right"
	plt.show(block=False)
	if saving:
		plt.savefig(f"figures/bad_regionPlot_{generate_file_label()}.pdf", bbox_inches='tight')
	return fig,ax


def generate_file_label():
	# weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_stamp = f"at_{timestamp}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nSelectedFac}of{nFac}"
	return trial_stamp





saving = True
yvals,rvals,uvals,prob_success,fstar,extra_info = solve_enumerative(obj_fn)
fig,ax = plot_region(f"{extra_info}",saving=saving)


# fig,ax = plot_region("(enumerative method)")


