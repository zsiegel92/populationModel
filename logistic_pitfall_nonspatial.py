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
# matplotlib.use('Qt5Agg')
# matplotlib.rcParams['text.usetex'] = True
saving = True
showing = True

# nIndiv = 100
# nFac = 10
theta_low = 0
theta_high = 1


# sum prob
theta = [theta_low,theta_high]
nIndiv = 2

beta = [-4,1.25,0.1] #[beta0, beta_theta >0, beta_r <0]

def probSuccess(ttheta,rr):
	return 1/(1+exp(-beta[0] - beta[1]*ttheta-beta[2]*rr))


def solve_casadi():
	opti = Opti()
	r = [opti.variable() for i in range(nIndiv)] # nIndiv
	opti.minimize(-sum([probSuccess(theta[i],r[i]) for i in range(nIndiv)])  )#maximize sum u
	opti.subject_to(sum([r[i] for i in range(nIndiv)]) == 1) #log prob(success)
	opti.subject_to([opti.bounded(0,r[i],1) for i in range(nIndiv)]) #log prob(success)
	p_options = {"expand":True}
	s_options = {}# {"max_iter": 100,'tol': 100}
	opti.solver('bonmin',p_options,s_options)
	sol = opti.solve()
	rvals = np.array([sol.value(r[i]) for i in range(nIndiv)])
	uvals = np.array([ probSuccess(theta[i],sol.value(r[i])) for i in range(nIndiv)])
	fstar = -1*sol.value(opti.f)
	print(f"Solved using CasADI")
	return rvals,uvals,fstar



def get_n_colors(number_desired_colors):
	if number_desired_colors < 10:
		default = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		colors = default[0:number_desired_colors]
	else:
		# https://matplotlib.org/tutorials/colors/colormaps.ht
		cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
		colors = [cmap(i) for i in range(number_desired_colors)]
	return colors

def plot_curves(saving=False):
	colors = get_n_colors(4)
	color_type_map = {ttheta : colors.pop() for ttheta in set(theta)}
	fig, ax = plt.subplots(figsize=(10,10))
	axin = ax.inset_axes([0.7,0.2,0.25,0.25])
	axin.patch.set_alpha(0.2) #fails
	axin.set_alpha(0.2) #fails
	# axin.set_axis_off()
	#[x0,y0,width,height],transform=ax.transData
	plt.xlabel("Resource Allocation")
	plt.ylabel("Individual Utility")
	plt.title(f"Unintended Triage: Nonspatial")
	big_r_range = np.linspace(-0.5,60,1000)
	small_r_range = np.linspace(-0.1,1.1,1000)
	
	for an_axis,r_range in zip((ax,axin),(big_r_range,small_r_range)):
		u = [[probSuccess(thetaval,rr) for rr in r_range] for thetaval in theta]
		for i,u_range in enumerate(u):
			an_axis.plot(r_range,u_range,c=color_type_map[theta[i]],label=f"Utility when $\\theta={theta[i]}$",zorder=0)
		for i in range(nIndiv):
			an_axis.scatter(rvals[i],uvals[i],color='black',label=f"Optimal Utility when $\\theta={theta[i]}$",zorder=1)


	# xin0,xin1 = axin.get_xbound()
	# yin0,yin1 = axin.get_ybound()
	# ax.indicate_inset(bounds=[xin0,yin0,xin1-xin0,yin1-yin0],inset_ax=axin,alpha=0.3)
	axin.set_alpha(0.2)
	ax.indicate_inset_zoom(axin,alpha=0.2)

	# plt.legend(loc="best")

	legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy() if "indicate_inset" not in artist.properties().get('label')} #Python 3.7
	plt.legend(legend_dict.values(),legend_dict.keys(),loc='best') #,loc="upper right"


	if showing:
		plt.show(block=False)
	if saving:
		plt.savefig(f"figures/logistic_problem_nonspatial_{generate_file_label()}.pdf", bbox_inches='tight')
	return fig,ax,axin


def generate_file_label():
	# weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_stamp = f"at_{timestamp}_beta_{beta_label}_nIndiv_{nIndiv}"
	return trial_stamp






rvals,uvals,fstar = solve_casadi()
fig,ax,axin = plot_curves(saving=saving)


# fig,ax = plot_region("(enumerative method)")


