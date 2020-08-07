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
nSelectedFac = 1
trial_label = "Logistic Pitfall"
indiv = np.array([[0.2,0.2],[0.2,0.8]])
fac = np.array([[0.2,0.1],[0.2,0.9]])
theta = [theta_high,theta_low]
nIndiv = indiv.shape[0]
nFac = fac.shape[0]
beta = [0,3,-10] #[beta0, beta_theta >0, beta_r <0]
facLabels = ["A","B"]

@setmeta(extra_info="Maximize Probability of Success of All Individuals")
def logPredProb(rr):
	u = [-log(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nIndiv)]
	return u


@setmeta(extra_info="Maximize $\\sum_{individuals} P(success)$")
def predProb(rr):
	u = [1/(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nIndiv)]
	return u


obj_fn=predProb




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

def plot_region_and_curves(title_extra="",saving=False,extra_info=""):
	colors = get_n_colors(6)
	color_type_map = {ttheta : colors.pop() for ttheta in set(theta)}
	facility_color_map = {yval : colors.pop() for yval in set(yvals)}
	facility_size_options = [5**2,9**2]
	facility_states = ["Omitted", "Selected"]

	gs = gridspec.GridSpec(nrows=3,ncols=1,height_ratios=[1,1,1])
	fig = plt.figure(figsize=(5,15))
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	ax3 = plt.subplot(gs[2])
	fig.subplots_adjust(hspace=0.2)
	decimal_formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

	## Plot Region
	title = f"Facilities and Individuals"
	if len(title_extra) > 0:
		title = f"{title}\n{title_extra}"
	ax1.set_title(title)
	ax1.set_xlim(0,1)
	ax1.set_xticks([])
	ax1.set_yticks([])

	def plotProbVsDistance():
		spchar=" "
		rspace = np.linspace(-0.1,0.75,num=1000)
		for theta_val in (theta_low,theta_high):
			uspace_type = [1/(1+exp(-beta[0] - beta[1]*theta_val - beta[2]*rval)) for rval in rspace]
			ax2.plot(rspace,uspace_type,color = color_type_map[theta_val],label=f"Utility{spchar}for Type {theta_val}",zorder=-1)
		low_index = theta.index(theta_low)
		high_index = theta.index(theta_high)
		# yvals_inverted = [0 if v==1 else 1 for v in yvals]
		# gp_inverted = [i for i,v in enumerate(yvals_inverted) if v==1]
		# xlist_inverted = [ gp_inverted[np.argmin([dist[i][j] for j in gp_inverted])] for i in range(nIndiv)]
		# r_inverted = [dist[i][xlist_inverted[i]] for i in range(nIndiv)]
		r_inverted = list(reversed(rvals))
		r_inverted_and_not = {1: rvals, 0: r_inverted}
		utility_inverted_and_not = {inversion : sum([1/(1+exp(-beta[0] - beta[1]*theta[ind] - beta[2]*rvals_possibly_inverted[ind])) for ind in range(nIndiv)]) for inversion,rvals_possibly_inverted in r_inverted_and_not.items()}

		for inversion,rvals_possibly_inverted in r_inverted_and_not.items():
			for ind in reversed(range(nIndiv)):
				ax2.scatter(rvals_possibly_inverted[ind],[1/(1+exp(-beta[0] - beta[1]*theta[ind] - beta[2]*rvals_possibly_inverted[ind]))],marker = "+",c=[facility_color_map[inversion]],s=[facility_size_options[inversion]],label=f"Utility from{spchar}{facility_states[inversion].lower()} facility (Sum: {utility_inverted_and_not[inversion]:.02})",zorder=1)
		# for ind in range(len(theta)):
		# 	ax2.scatter(r_inverted[ind],[1/(1+exp(-beta[0] - beta[1]*theta[ind] - beta[2]*r_inverted[ind]))],marker = "+",c=[facility_color_map[0]],s=[facility_size_options[0]],label=f"Utility from {facility_states[0]} facility")
		legend_dict = {artist.properties().get('label') : artist for artist in ax2.collections.copy() + ax2.lines.copy() if "no_legend" not in artist.properties().get('label')}
		# ax2.legend(legend_dict.values(),legend_dict.keys(),loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(legend_dict),fontsize='small')
		ax2.legend(legend_dict.values(),legend_dict.keys(),loc='best',fontsize='small')
		ax2.set_xlabel("distance from nearest selected facility")
		ax2.set_ylabel("$P(Success)$")
		ax2.set_title("$P(success)$ vs Distance Differs by Type")
	plotProbVsDistance()


	def plotLogProbVsDistance():
		spchar=" "
		rspace = np.linspace(-0.1,0.75,num=1000)
		for theta_val in (theta_low,theta_high):
			uspace_type = [-log(1+exp(-beta[0] - beta[1]*theta_val - beta[2]*rval)) for rval in rspace]
			ax3.plot(rspace,uspace_type,color = color_type_map[theta_val],label=f"Utility{spchar}for Type {theta_val}",zorder=-1)
		low_index = theta.index(theta_low)
		high_index = theta.index(theta_high)
		# yvals_inverted = [0 if v==1 else 1 for v in yvals]
		# gp_inverted = [i for i,v in enumerate(yvals_inverted) if v==1]
		# xlist_inverted = [ gp_inverted[np.argmin([dist[i][j] for j in gp_inverted])] for i in range(nIndiv)]
		# r_inverted = [dist[i][xlist_inverted[i]] for i in range(nIndiv)]
		r_inverted = list(reversed(rvals))
		r_inverted_and_not = {1: rvals, 0: r_inverted}
		utility_inverted_and_not = {inversion : sum([-log(1+exp(-beta[0] - beta[1]*theta[ind] - beta[2]*rvals_possibly_inverted[ind])) for ind in range(nIndiv)]) for inversion,rvals_possibly_inverted in r_inverted_and_not.items()}

		for inversion,rvals_possibly_inverted in r_inverted_and_not.items():
			for ind in reversed(range(nIndiv)):
				ax3.scatter(rvals_possibly_inverted[ind],[-log(1+exp(-beta[0] - beta[1]*theta[ind] - beta[2]*rvals_possibly_inverted[ind]))],marker = "+",c=[facility_color_map[inversion]],s=[facility_size_options[inversion]],label=f"Utility from{spchar}{facility_states[inversion].lower()} facility (Sum: {utility_inverted_and_not[inversion]:.02})",zorder=1)
		# for ind in range(len(theta)):
		# 	ax3.scatter(r_inverted[ind],[1/(1+exp(-beta[0] - beta[1]*theta[ind] - beta[2]*r_inverted[ind]))],marker = "+",c=[facility_color_map[0]],s=[facility_size_options[0]],label=f"Utility from {facility_states[0]} facility")
		legend_dict = {artist.properties().get('label') : artist for artist in ax3.collections.copy() + ax3.lines.copy() if "no_legend" not in artist.properties().get('label')}
		# ax3.legend(legend_dict.values(),legend_dict.keys(),loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(legend_dict),fontsize='small')
		ax3.legend(legend_dict.values(),legend_dict.keys(),loc='best',fontsize='small')
		ax3.set_xlabel("distance from nearest selected facility")
		ax3.set_ylabel("$log[P(Success)]$")
		ax3.set_title("$log[P(success)]$ vs Distance Differs by Type")
	plotLogProbVsDistance()

	## Plot Objectives
	def plotObjectives():
		obj_scatter = ax1.scatter(indiv[:,0],indiv[:,1],marker="o",s=14**2,lw=2,alpha=0.5,c=prob_success,cmap="cividis",label="no_legend")
		obj_scatter.set_facecolor('none')
		cbar = fig.colorbar(obj_scatter,ax=ax1)
		cbar.set_clim(0,1)
		cbar.ax.set_ylabel('P(success)')
	plotObjectives()

	## Plot Individuals
	labelIndividuals = "Individual"
	def plotIndividuals():
		individual_colors = [color_type_map[ttheta] for ttheta in theta]
		individual_labels = [f"{labelIndividuals} type {ttheta}" for coords,ttheta in zip(indiv,theta)]
		for i,coords in enumerate(indiv):
			ax1.scatter(*coords,marker = "*",c=[individual_colors[i]],label=individual_labels[i])
	plotIndividuals()

	## Plot Facilities
	labelFacility = "Facility"
	def plotFacilities():
		facility_sizes = [facility_size_options[yval] for yval in yvals]
		facility_colors = [facility_color_map[yval] for yval in yvals]
		facility_labels = [f"{facility_states[yval]} {labelFacility}" for yval in yvals]
		for i, coords in enumerate(fac):
			ax1.scatter(*coords,marker = "+",c=[facility_colors[i]],s=[facility_sizes[i]],label=facility_labels[i])
		for i,lab in enumerate(facLabels):
			ax1.text(fac[i][0]+.01,fac[i][1]-.01,lab,fontsize=10)
	plotFacilities()

	## Create Legend
	# legend_dict = {legendTitle : artist for artist in ax1.collections.copy() + ax1.lines.copy() if "no_legend" not in (legendTitle:=artist.properties().get('label'))} #Python 3.8
	legend_dict = {artist.properties().get('label') : artist for artist in ax1.collections.copy() + ax1.lines.copy() if "no_legend" not in artist.properties().get('label')} #Python 3.7
	# ax1.legend(legend_dict.values(),legend_dict.keys(),loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(legend_dict),fontsize='small') #,loc="upper right"
	ax1.legend(legend_dict.values(),legend_dict.keys(),loc='best',fontsize='small')
	# fig.tight_layout()
	plt.show(block=False)
	if saving:
		plt.savefig(f"figures/logistic_problem_{generate_file_label()}.pdf", bbox_inches='tight')


	return fig,ax1,ax2,ax3


def generate_file_label():
	# weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_stamp = f"at_{timestamp}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nSelectedFac}of{nFac}"
	return trial_stamp





saving = True
yvals,rvals,uvals,prob_success,fstar,extra_info = solve_enumerative(obj_fn)
fig,ax1,ax2,ax3 = plot_region_and_curves(f"{extra_info}",saving=saving)


# fig,ax = plot_region("(enumerative method)")


