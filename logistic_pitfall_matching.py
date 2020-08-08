from casadi import Opti,log,exp
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.sparse import coo_matrix as sparse
from subset_helper import bax
from function_helpers import setmeta
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True
saving = False
showing = True

# nIndiv = 100
# nFac = 10
theta_low = 0
theta_high = 1
nIndiv = 100
nFac = 80
nEdgePerIndiv = 5
nEdges = nIndiv * nEdgePerIndiv
beta = [-4,1.25,0.1] #[beta0, beta_theta >0, beta_r <0]

def edge_val(theta_val,fac_val):
	return 1/(1+exp(-beta[0] - beta[1]*theta_val-beta[2]*fac_val))

edges_list_of_lists = [sorted(np.random.choice(range(nFac),nEdgePerIndiv)) for i in range(nIndiv)]
edges = [(i,e) for i in range(nIndiv) for e in edges_list_of_lists[i]]
# edges = [(i,e) for i in range(nIndiv) for e in sorted(np.random.choice(range(nFac),nEdgePerIndiv,replace=False))]
fac_vals = [np.random.rand() for i in range(nFac)]
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]

edge_weights = [edge_val(theta[indiv_ind],fac_vals[fac_ind]) for (indiv_ind,fac_ind) in edges]
weights_mat = sparse((edge_weights,list(zip(*edges)))).toarray()

edge_index = {edge: ind for ind,edge in enumerate(edges)}

used_fac = sorted(list(set(e for (i,e) in edges)))

# @setmeta(name="$P(Success)$",axes_inset=[0.7,0.05,0.25,0.25])
# def probSuccess(ttheta,rr):
# 	return 1/(1+exp(-beta[0] - beta[1]*ttheta-beta[2]*rr))

# @setmeta(name="$\\log \\left(P(Success)\\right)$",axes_inset=[0.7,0.05,0.25,0.25])
# def logProbSuccess(ttheta,rr):
# 	return -log(1+exp(-beta[0] - beta[1]*ttheta-beta[2]*rr))

def solve_casadi():
	opti = Opti()
	x = [opti.variable() for tup in edges] # nIndiv

	opti.minimize(-sum([edge_weight * xi for (edge_weight,xi) in zip(edge_weights,x)])  )#maximize sum u

	# for ind in range(nIndiv):
		# opti.subject_to(sum([x[edge_index[(ind,fac)]] for fac in edges_list_of_lists[ind]])<=1)
	opti.subject_to([ sum([x[edge_index[(ind,fac)]] for fac in edges_list_of_lists[ind]])<=1 for ind in range(nIndiv) ]) #individuals have no more than one facility
	# for fac in range(nFac):
	# 	if fac in used_fac:
	# 		opti.subject_to(sum([x[edge_index[(ind,fac)]] for ind in range(nIndiv) if fac in edges_list_of_lists[ind]])<=1)
	opti.subject_to([ sum([x[edge_index[(ind,fac)]] for ind in range(nIndiv) if fac in edges_list_of_lists[ind]])<=1 for fac in used_fac ]) #facilities have no more than one individual
	opti.subject_to([opti.bounded(0,x[i],1) for i in range(nEdges)])
	
	# discrete = [True for i in range(nEdges)]
	p_options = {"expand":True} # , "discrete":discrete
	s_options = {}# {"max_iter": 100,'tol': 100}
	opti.solver('bonmin',p_options,s_options)
	sol = opti.solve()
	xvals = np.array([sol.value(x[i]) for i in range(nEdges)])

	edgeVals = {edge : xvals[edge_index[edge]] for edge in edges}
	indVals = [max([edgeVals[edge] for edge in edgeVals if edge[0] == ind]) for ind in range(nIndiv)]
	# uvals = np.array([ obj_fn(theta[i],sol.value(r[i])) for i in range(nIndiv)])
	#TODO: get each individual's utility - the edge weight of the 1-value variable
	fstar = -1*sol.value(opti.f)
	print(f"Solved using CasADI")
	return xvals,fstar,edgeVals,indVals,



# def get_n_colors(number_desired_colors):
# 	if number_desired_colors < 10:
# 		default = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# 		colors = default[0:number_desired_colors]
# 	else:
# 		# https://matplotlib.org/tutorials/colors/colormaps.ht
# 		cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
# 		colors = [cmap(i) for i in range(number_desired_colors)]
# 	return colors

# def plot_curves(saving=False):
# 	colors = get_n_colors(4)
# 	color_type_map = {ttheta : colors.pop() for ttheta in set(theta)}

# 	nPlots = len(solutions)
# 	gs = gridspec.GridSpec(nrows=nPlots,ncols=1,height_ratios=[1 for sol in solutions])
# 	fig = plt.figure(figsize=(5,5*nPlots))
# 	axes = [plt.subplot(gs[i]) for i in range(nPlots)]
# 	inset_axes = []
# 	fig.subplots_adjust(hspace=0.2)
# 	# fig, ax = plt.subplots(figsize=(10,10))
# 	plt.title(f"Unintended Triage: Nonspatial")
# 	for indPlot,obj_fn in enumerate(solutions):
# 		ax = axes[indPlot]
# 		axin = ax.inset_axes(obj_fn.axes_inset)
# 		inset_axes.append(axin)
# 		#[x0,y0,width,height],transform=ax.transData
# 		ax.set_xlabel("Resource Allocation")
# 		ax.set_ylabel(f"Individual {obj_fn.name}")
# 		plotTitle = ("Unintended Triage: Nonspatial\n" if indPlot ==0 else "") + f"Optimizing $\\sum${obj_fn.name}"
# 		ax.set_title(plotTitle)
# 		big_r_range = np.linspace(-0.5,40,1000)
# 		small_r_range = np.linspace(-0.1,1.1,1000)


# 		for an_axis,r_range in zip((ax,axin),(big_r_range,small_r_range)):
# 			u = [[obj_fn(thetaval,rr) for rr in r_range] for thetaval in theta]
# 			for i,u_range in enumerate(u):
# 				an_axis.plot(r_range,u_range,c=color_type_map[theta[i]],label=f"Utility when $\\theta={theta[i]}$",zorder=0)
# 			for i in range(nIndiv):
# 				an_axis.scatter(solutions[obj_fn]['rvals'][i],solutions[obj_fn]['uvals'][i],color='black',label=f"Optimal Allocation when $\\theta={theta[i]}$",zorder=1)


# 		# xin0,xin1 = axin.get_xbound()
# 		# yin0,yin1 = axin.get_ybound()
# 		# ax.indicate_inset(bounds=[xin0,yin0,xin1-xin0,yin1-yin0],inset_ax=axin,alpha=0.3)
# 		# axin.set_alpha(0.2)
# 		# axin.set_xticks(fontsize=9)
# 		axin.set_xticklabels(axin.get_xticks(),fontsize=6)
# 		axin.set_yticklabels(axin.get_yticks(),fontsize=6)
# 		ax.indicate_inset_zoom(axin,alpha=0.2)

# 		# plt.legend(loc="best")

# 		legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy() if "indicate_inset" not in artist.properties().get('label')} #Python 3.7
# 		ax.legend(legend_dict.values(),legend_dict.keys(),loc='best',fontsize='x-small') #,loc="upper right"


# 	if showing:
# 		plt.show(block=False)
# 	if saving:
# 		plt.savefig(f"figures/logistic_problem_nonspatial_{generate_file_label()}.pdf", bbox_inches='tight')
# 	return fig,axes,inset_axes


# def generate_file_label():
# 	# weights_label = "_".join(map(str,weights))
# 	beta_label = "_".join(map(str,beta))
# 	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
# 	trial_stamp = f"at_{timestamp}_beta_{beta_label}_nIndiv_{nIndiv}"
# 	return trial_stamp


#
xvals,fstar,edgeVals,indVals = solve_casadi()

# solutions = {obj_fn : dict(zip(("rvals","uvals","fstar"),solve_casadi(obj_fn))) for obj_fn in (probSuccess,logProbSuccess)}

# # for obj_fn in (probSuccess,logProbSuccess):
# # 	rvals,uvals,fstar = solve_casadi(obj_fn)
# fig,ax,axin = plot_curves(saving=saving)


# fig,ax = plot_region("(enumerative method)")


