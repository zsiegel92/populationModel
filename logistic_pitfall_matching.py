from casadi import Opti,log,exp
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
# from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import coo_matrix as sparse
from subset_helper import bax
from function_helpers import setmeta
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True
saving = True
showing = True

# nIndiv = 100
# nFac = 10
theta_low = 0
theta_high = 1
nIndiv = 20
nFac = 16
nEdgePerIndiv = 2
nEdges = nIndiv * nEdgePerIndiv
beta = [-4,1.25,0.1] #[beta0, beta_theta >0, beta_r <0]

def edge_val(theta_val,fac_val):
	return 1/(1+exp(-beta[0] - beta[1]*theta_val-beta[2]*fac_val))

@setmeta(title="P(Success)")
def raw_utility(utility):
	return utility

@setmeta(title="Log(P(Success))")
def log_utility(utility):
	return log(utility)
obj_fns = (raw_utility,log_utility)

edges_list_of_lists = [sorted(np.random.choice(range(nFac),nEdgePerIndiv,replace=False)) for i in range(nIndiv)]
edges = [(i,e) for i in range(nIndiv) for e in edges_list_of_lists[i]]
# edges = [(i,e) for i in range(nIndiv) for e in sorted(np.random.choice(range(nFac),nEdgePerIndiv,replace=False))]
fac_vals = sorted([np.random.rand() for i in range(nFac)])
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
edge_weights = [edge_val(theta[indiv_ind],fac_vals[fac_ind]) for (indiv_ind,fac_ind) in edges]
weights_mat = sparse((edge_weights,list(zip(*edges)))).toarray()
edge_index = {edge: ind for ind,edge in enumerate(edges)}
used_fac = sorted(list(set(e for (i,e) in edges)))
MMM = nIndiv*max([abs(log(ew)) for ew in edge_weights]) +1000
# @setmeta(name="$P(Success)$",axes_inset=[0.7,0.05,0.25,0.25])
# def probSuccess(ttheta,rr):
# 	return 1/(1+exp(-beta[0] - beta[1]*ttheta-beta[2]*rr))

# @setmeta(name="$\\log \\left(P(Success)\\right)$",axes_inset=[0.7,0.05,0.25,0.25])
# def logProbSuccess(ttheta,rr):
# 	return -log(1+exp(-beta[0] - beta[1]*ttheta-beta[2]*rr))

def solve_casadi():
	all_xvals = {}
	all_edgeVals = {}
	all_indVals = {}
	all_edgeUtility = {}
	all_indUtility = {}

	for obj_fn in obj_fns:
		opti = Opti()
		x = [opti.variable() for tup in edges] # nIndiv
		discrete = [True for tup in edges]
		u = [ 1/(1+exp(-beta[0] - beta[1]*theta[i] - beta[2] * sum([fac_vals[jj] * x[edge_ind] for edge_ind,(ii,jj) in enumerate(edges) if ii == i ]))) for i in range(nIndiv)]
		uu = [obj_fn(ui) for ui in u]
		opti.minimize(-sum(uu))


		opti.subject_to([ sum([x[edge_index[(ind,fac)]] for fac in edges_list_of_lists[ind]])<=1 for ind in range(nIndiv) ]) #individuals have no more than one facility
		opti.subject_to([ sum([x[edge_index[(ind,fac)]] for ind in range(nIndiv) if fac in edges_list_of_lists[ind]])<=1 for fac in used_fac ]) #facilities have no more than one individual
		opti.subject_to([opti.bounded(0,x[i],1) for i in range(nEdges)])
		p_options = {"expand":True, "discrete":discrete} #
		s_options = {}# {"max_iter": 100,'tol': 100}
		opti.solver('bonmin',p_options,s_options)
		sol = opti.solve()
		xvals= np.array([sol.value(x[i]) for i in range(nEdges)])
		edgeVals= {edge : xvals[edge_index[edge]] for edge in edges}
		indVals= [max([edgeVals[edge] for edge in edgeVals if edge[0] == ind]) for ind in range(nIndiv)]

		u = [ 1/(1+exp(-beta[0] - beta[1]*theta[i] - beta[2] * sum([fac_vals[jj] * xvals[edge_ind] for edge_ind,(ii,jj) in enumerate(edges) if ii == i ]))) for i in range(nIndiv)]
		indUtility = [obj_fn(ui) for ui in u]


		edgeUtility= {(ii,jj) : indUtility[ii] if withinTol(edgeVals[(ii,jj)],1) else -np.inf for edge_ind, (ii,jj) in enumerate(edges)}


		all_xvals[obj_fn] = xvals
		all_edgeVals[obj_fn] = edgeVals
		all_indVals[obj_fn] = indVals
		all_edgeUtility[obj_fn] = edgeUtility
		all_indUtility[obj_fn] = indUtility
	# fstar = -1*sol.value(opti.f)
	print(f"Solved using CasADI")
	return all_xvals,all_edgeVals,all_indVals,all_edgeUtility,all_indUtility,



def get_n_colors(number_desired_colors):
	if number_desired_colors < 10:
		default = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		colors = default[0:number_desired_colors]
	else:
		cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
		colors = [cmap(i) for i in range(number_desired_colors)]
	return colors

def withinTol(a,b,tol=.001):
	return abs(a-b) < tol

def plot_graph(saving=False):


	colors = get_n_colors(4)
	color_type_map = {ttheta : colors.pop() for ttheta in set(theta)}
	facColor = colors.pop()
	noMatchColor = colors.pop()
	unused_edge_color = noMatchColor#colors.pop()
	indivX = 0.1*np.ones(nIndiv)
	indivY = np.linspace(0.1,0.9,nIndiv)

	facX = 0.9 * np.ones(nFac)
	facY = np.linspace(0.1,0.9,nFac)

	nPlots = len(obj_fns)
	gs = gridspec.GridSpec(nrows=nPlots,ncols=1,height_ratios=np.ones(nPlots))
	# fig.subplots_adjust(hspace=0.2)
	fig = plt.figure(figsize=(5,5*nPlots))
	axes = [plt.subplot(gs[i]) for i in range(nPlots)]
	for ax,obj_fn in zip(axes,obj_fns):
		# xv = xvals[obj_fn]
		edgeV = edgeVals[obj_fn]
		indV = indVals[obj_fn]
		edgeU = edgeUtility[obj_fn]
		indU = indUtility[obj_fn]
		ax.set_title(f"Maximum Weighted Matching\nMaximize $\\sum${obj_fn.title}")
		ax.set_axis_off()
		base_size=36
		for theta_val in set(theta):
			ax.scatter([indivX[i] for i in range(nIndiv) if theta[i]==theta_val],[indivY[i] for i in range(nIndiv) if theta[i]==theta_val],c=[color_type_map[theta[i]] for i in range(nIndiv) if theta[i]==theta_val],label=f"Individual Type {theta_val}",s=base_size,zorder=0)
		ax.scatter(facX,facY,c=facColor,s=base_size*2*np.array(fac_vals),label=f"Facility (size = value)",zorder=0)
		edgeUVals = np.array(list(edgeU.values()))
		minEdgeVal = min(edgeUVals[edgeUVals!= -np.inf])
		maxEdgeVal = max(edgeUVals)
		cmap = plt.cm.get_cmap('cividis')
		norm = Normalize(vmin=minEdgeVal, vmax=maxEdgeVal)
		sm = ScalarMappable(norm=norm, cmap=cmap)
		# sm.set_array([])
		for (ii,jj) in edges:
			#TODO: colormap for edge colors!
			if withinTol(edgeV[(ii,jj)],1):
				ax.plot(
					[indivX[ii],facX[jj]],[indivY[ii],facY[jj]],
					alpha=1,
					color=sm.to_rgba(edgeU[(ii,jj)]),
					zorder=-1,
					label="no_legend")#,label="Utility to Individual"
			else:
				ax.plot(
					[indivX[ii],facX[jj]],[indivY[ii],facY[jj]],
					alpha=0.1,
					color=unused_edge_color,
					zorder=-10,
					label="Unused Edge")#,label="Utility to Individual"
		for ii,u in enumerate(indU):
			if withinTol(indV[ii],0):
				xx= indivX[ii]
				yy = indivY[ii]
				xoffset = .05
				ax.plot(
				        [xx,xx-xoffset],[yy,yy],
				        alpha=0.5,
				        color=sm.to_rgba(indU[ii]),
				        zorder=-1,
				        label="no_legend",
				        linestyle="-",
				        markersize=4,
				        marker="*",
				        markerfacecolor=noMatchColor,
				        markeredgecolor=noMatchColor
				        )
				a_no_match_color = sm.to_rgba(indU[ii])
		cbar = fig.colorbar(sm,ax=ax,drawedges=False)
		cbar.ax.set_ylabel(f'Individual {obj_fn.title}')
		legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy() if "no_legend" not in artist.properties().get('label')}
		mean_color = sm.to_rgba(np.mean(edgeUVals[edgeUVals!= -np.inf]))
		noMatchFake = matplotlib.lines.Line2D([0], [0],
		                                   label=f"Unmatched Individual",
		                                   color=mean_color,
		                                   alpha=0.8,
		                                   linestyle='-',
		                                   markersize=4,
		                                   marker="*",
		                                   markerfacecolor=noMatchColor,
		                                   markeredgecolor=noMatchColor
		                                   )
		fake_dict = {noMatchFake.get_label() : noMatchFake}
		legend_dict = {**legend_dict,**fake_dict}




		ax.legend(legend_dict.values(),legend_dict.keys(),loc='best',fontsize='x-small') #,loc="upper right"

	if showing:
		plt.show(block=False)
	if saving:
		plt.savefig(f"figures/logistic_matching_{generate_file_label()}.pdf", bbox_inches='tight')
	return fig,ax


def generate_file_label():
	# weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_stamp = f"at_{timestamp}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nFac}_nEdgePerIndiv_{nEdgePerIndiv}"
	return trial_stamp



xvals,edgeVals,indVals,edgeUtility,indUtility = solve_casadi()
fig,ax = plot_graph(saving=saving)
# solutions = {obj_fn : dict(zip(("rvals","uvals","fstar"),solve_casadi(obj_fn))) for obj_fn in (probSuccess,logProbSuccess)}

# # for obj_fn in (probSuccess,logProbSuccess):
# # 	rvals,uvals,fstar = solve_casadi(obj_fn)
# fig,ax,axin = plot_curves(saving=saving)


# fig,ax = plot_region("(enumerative method)")


