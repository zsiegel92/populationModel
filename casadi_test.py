from casadi import Opti,log,exp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True
number_desired_colors = 6
cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
colors = [cmap(i) for i in range(number_desired_colors)]




sizeRegion = 1

nIndiv = 25
nFac = 5
nSelectedFac = 2

indiv = np.random.uniform(0,sizeRegion,(nIndiv, 2))
# theta = np.random.uniform(0,1,(nIndiv,1))
theta = np.array([np.random.choice([0,1]) for i in range(nIndiv)])
fac = np.random.uniform(0,sizeRegion,(nFac, 2))

dist = np.array([[ (indiv[i,0] - fac[j,0])**2 + (indiv[i,1] - fac[j,1])**2 for j in range(0,nFac)] for i in range(0,nIndiv)]) # nIndiv X nFac

beta = [0,10,-1] #[beta0, beta_theta >0, beta_r <0]



opti = Opti()


x = [[opti.variable() for j in range(nFac)] for i in range(nIndiv)] # nIndiv X nFac
y = [opti.variable() for j in range(nFac)]
r = [opti.variable() for i in range(nIndiv)]
u = [opti.variable() for i in range(nIndiv)]

discrete = []
discrete += [True  for j in range(nFac) for i in range(nIndiv)] #x variables
discrete += [True  for j in range(nFac)] #y variables
discrete += [False for i in range(nIndiv)] #r variables
discrete += [False for i in range(nIndiv)] #u variables



opti.minimize(-sum(u))#maximize sum u
# opti.subject_to([u[i] == -r[i] for i in range(nIndiv)])
opti.subject_to([u[i] == -log(1+exp(-beta[0] - beta[1]*theta[i] - beta[2]*r[i])) for i in range(nIndiv)])
opti.subject_to([r[i] >= dist[i,j]*x[i][j] for i in range(nIndiv) for j in range(nFac)])
opti.subject_to([ sum(x[i]) == 1 for i in range(nIndiv) ])
opti.subject_to([ x[i][j] <= y[j] for i in range(nIndiv) for j in range(nFac)])
opti.subject_to(sum(y) <= nSelectedFac)



opti.subject_to([ opti.bounded(0,y[j],1) for j in range(nFac)])
opti.subject_to([ opti.bounded(0,x[i][j],1) for i in range(nIndiv) for j in range(nFac)])


# opti.callback(@(i): print(f"At iteration {i}"))


p_options = {"discrete":discrete,"expand":True}
s_options = {"max_iter": 100}
opti.solver('bonmin',p_options,s_options)
sol = opti.solve()

# print(sol.value(x),sol.value(y),sol.value(z))
xvals = np.array([[int(sol.value(x[i][j])) for j in range(nFac)] for i in range(nIndiv)])
yvals = np.array([int(sol.value(y[j])) for j in range(nFac)])
rvals = np.array([sol.value(r[i]) for i in range(nIndiv)])
uvals = np.array([sol.value(u[i]) for i in range(nIndiv)])
fstar = sol.value(opti.f)



def plot_region():
	fig, ax = plt.subplots(figsize=(5,5))
	plt.axis('off')
	plt.title(f"Region with facilities and individuals\n(higher type -> success more likely)")

	## Plot Region
	# labelRegion="Region"
	region = patches.Rectangle((0,0),sizeRegion,sizeRegion,linewidth=1,edgecolor=colors.pop(),facecolor="none")
	ax.add_patch(region)

	# plt.plot([*r.getExteriorCoords()],color=colors.pop(),label=labelRegion)
	## Plot Individuals
	labelIndividuals = "Individual"
	def plotIndividuals():
		# plt.scatter(*pop.getIndividualCoordinates(),marker="*",c="blue",label="Individual")
		color_weight_map = {ttheta : colors.pop() for ttheta in set(theta)}
		individual_colors = [color_weight_map[ttheta] for ttheta in theta]
		individual_labels = [f"{labelIndividuals} type {ttheta}" for coords,ttheta in zip(indiv,theta)]
		for i,coords in enumerate(indiv):
			plt.scatter(*coords,marker = "*",c=[individual_colors[i]],label=individual_labels[i])
	plotIndividuals()
	## Plot Facilities
	labelFacility = "Facility"
	def plotFacilities():
		facility_color_map = {yval : colors.pop() for yval in set(yvals)}
		facility_colors = [facility_color_map[yval] for yval in yvals]
		facility_states = ["Omitted", "Selected"]
		facility_labels = [f"{facility_states[yval]} {labelFacility}" for yval in yvals]
		for i, coords in enumerate(fac):
			plt.scatter(*coords,marker = "+",c=[facility_colors[i]],label=facility_labels[i])
			# plt.scatter(*r.getFacilityLocations(),marker="+",c=[colors.pop()],label=labelFacility)
	plotFacilities()
	## Create Legend
	legend_dict = {artist.properties().get('label') : artist for artist in ax.collections.copy() + ax.lines.copy()}
	plt.legend(legend_dict.values(),legend_dict.keys(),loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(legend_dict),fontsize='xx-small') #,loc="upper right"

	# plt.tight_layout()
	plt.show(block=False)
	return fig,ax

fig,ax = plot_region()




