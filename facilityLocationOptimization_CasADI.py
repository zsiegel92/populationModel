from casadi import Opti,log,exp
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from subset_helper import bax
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True


sizeRegion = 1

nIndiv = 100
nFac = 10
nSelectedFac = 5

indiv = np.random.uniform(0,sizeRegion,(nIndiv, 2))
# theta = np.random.uniform(0,1,(nIndiv,1))
# theta = np.array([np.random.choice([0,1]) for i in range(nIndiv)])
theta_low = 0
theta_high = 1
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
np.random.shuffle(theta)
fac = np.random.uniform(0,sizeRegion,(nFac, 2))

dist = np.array([[ (indiv[i,0] - fac[j,0])**2 + (indiv[i,1] - fac[j,1])**2 for j in range(0,nFac)] for i in range(0,nIndiv)]) # nIndiv X nFac

beta = [0,1,-10] #[beta0, beta_theta >0, beta_r <0]


def solve_casadi():
	opti = Opti()
	x = [[opti.variable() for j in range(nFac)] for i in range(nIndiv)] # nIndiv X nFac
	y = [opti.variable() for j in range(nFac)]
	r = [opti.variable() for i in range(nIndiv)]
	u = [opti.variable() for i in range(nIndiv)]
	discrete = []
	discrete += [False  for j in range(nFac) for i in range(nIndiv)] #x variables - will be binary without integer constraint
	discrete += [True  for j in range(nFac)] #y variables
	discrete += [False for i in range(nIndiv)] #r variables
	discrete += [False for i in range(nIndiv)] #u variables
	opti.minimize(-sum(u))#maximize sum u
	opti.subject_to([u[i] == -log(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*r[i])) for i in range(nIndiv)]) #log prob(success)
	# opti.subject_to([u[i] == 1/(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*r[i])) for i in range(nIndiv)]) #prob(success)
	opti.subject_to([r[i] >= sum([dist[i,j]*x[i][j] for j in range(nFac)]) for i in range(nIndiv) ])
	opti.subject_to([ sum(x[i]) == 1 for i in range(nIndiv) ])
	opti.subject_to([ x[i][j] <= y[j] for i in range(nIndiv) for j in range(nFac)])
	opti.subject_to(sum(y) <= nSelectedFac)
	opti.subject_to([ opti.bounded(0,y[j],1) for j in range(nFac)])
	opti.subject_to([ opti.bounded(0,x[i][j],1) for i in range(nIndiv) for j in range(nFac)])
	p_options = {"discrete":discrete,"expand":True}
	s_options = {"max_iter": 100,'tol': 100}
	opti.solver('bonmin',p_options,s_options)
	sol = opti.solve()
	# possibilities for tolerance in s_options (https://web.casadi.org/python-api/):
	#only 'tol' works...
	# for key in ['tol','boundTolerance','epsIterRef','terminationTolerance','abstol','opttol']:
	# 	try:
	# 		s_options = {"max_iter": 100,key: .1}
	# 		opti.solver('bonmin',p_options,s_options)
	# 		sol = opti.solve()
	# 		print(f"NOTE: '{key}' works in s_options!!")
	# 	except Exception as e:
	# 		print(e)
	# 		print(f"NOTE: '{key}' is not a valid s_option!")

	xvals = np.round(np.array([[sol.value(x[i][j]) for j in range(nFac)] for i in range(nIndiv)])).astype(int)
	yvals = np.array([sol.value(y[j]) for j in range(nFac)]).astype(int)
	rvals = np.array([sol.value(r[i]) for i in range(nIndiv)])
	uvals = np.array([sol.value(u[i]) for i in range(nIndiv)])
	# prob_success = np.array([1/(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*rvals[i])) for i in range(nIndiv)])
	prob_success = np.exp(uvals)
	fstar = sol.value(opti.f)
	print(f"Solved using CasADI")
	return xvals,yvals,rvals,uvals,prob_success,fstar



def solve_enumerative():
	bbax=bax()
	yvals = np.zeros((nFac),dtype=int)
	rvals = np.zeros((nIndiv))
	uvals = np.full((nIndiv),-np.inf)

	for ind, gp in enumerate(bbax.bax_gen(nFac,nSelectedFac)):
		gp = gp.toList()
		xlist = [ gp[np.argmin([dist[i][j] for j in gp])] for i in range(nIndiv)]
		r = [dist[i][xlist[i]] for i in range(nIndiv)]
		u = [-log(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*r[i])) for i in range(nIndiv)]
		if sum(u) > sum(uvals):
			uvals = u
			rvals = r
			yvals = np.zeros((nFac),dtype=int)
			yvals[gp] = 1

	fstar = sum(uvals)
	prob_success = np.exp(uvals)
	print(f"Solved enumeratively")
	return yvals,rvals,uvals,prob_success,fstar


def plot_region(title_extra=""):
	number_desired_colors = 6
	cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
	colors = [cmap(i) for i in range(number_desired_colors)]
	fig, ax = plt.subplots(figsize=(10,10))
	plt.axis('off')
	title = f"Region with facilities and individuals\n(higher type -> success more likely)"
	if len(title_extra) > 0:
		title = f"{title}\n{title_extra}"
	plt.title(title)

	## Plot Region
	region = patches.Rectangle((0,0),sizeRegion,sizeRegion,linewidth=1,edgecolor=colors.pop(),facecolor="none")
	ax.add_patch(region)

	## Plot Objectives
	def plotObjectives():
		# cm = plt.cm.get_cmap('winter')
		obj_scatter = plt.scatter(indiv[:,0],indiv[:,1],marker="o",s=14**2,lw=2,alpha=0.5,c=prob_success,cmap="winter",label="no_legend")
		obj_scatter.set_facecolor('none')
		cbar = plt.colorbar(obj_scatter)
		plt.clim(0,1)
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
	plotFacilities()

	## Create Legend
	# legend_dict = {legendTitle : artist for artist in ax.collections.copy() + ax.lines.copy() if "no_legend" not in (legendTitle:=artist.properties().get('label'))} #Python 3.8
	legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy() if "no_legend" not in artist.properties().get('label')} #Python 3.7
	plt.legend(legend_dict.values(),legend_dict.keys(),loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(legend_dict),fontsize='small') #,loc="upper right"
	plt.show(block=False)
	return fig,ax


def generate_file_label():
	# weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_label = f"at_{timestamp}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nSelectedFac}of{nFac}"
	return trial_label


# xvals2,yvals2,rvals2,uvals2,fstar2 = solve_casadi()
# xvals,yvals,rvals,uvals,fstar = xvals2,yvals2,rvals2,uvals2,fstar2
xvals, yvals,rvals,uvals,prob_success,fstar = solve_casadi()
# yvals,rvals,uvals,prob_success,fstar = solve_enumerative()
fig,ax = plot_region("(CasADI method)")

plt.savefig(f"figures/regionPlot_{generate_file_label()}.pdf", bbox_inches='tight')
# fig,ax = plot_region("(enumerative method)")


