import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from subset_helper import bax
from objective_factories import sum_log_prob_weighted_factory, sum_prob_weighted_factory,sum_distance_weighted_factory,prob_success
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True


nIndiv = 50
nFac = 16
nSelectedFac = 8
nTrials = 200

sizeRegion = 1


theta_low = 0
theta_high = 1
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
np.random.shuffle(theta)

beta = [0,0.2,-1] #[beta0, beta_theta >0, beta_r <0]

weights = [1,5,1000] #should all be >= 1
objective_functions = [sum_log_prob_weighted_factory(w,beta,theta) for w in weights] + [sum_prob_weighted_factory(w,beta,theta) for w in weights] + [sum_distance_weighted_factory(w,theta) for w in weights]
fairness_ratings = [10**10 + weight for weight in weights]+ [10**5 + weight for weight in weights] + weights #arbitrary, but roughly increasing with fairness-inducing tendency
fairness_strengths = dict(zip(objective_functions,fairness_ratings)) # to create a consistent order in plot legends

def solve_multiple_enumerative():
	bbax=bax()


	all_yvals = {fn : [] for fn in objective_functions}
	all_rvals = {fn : [] for fn in objective_functions}
	all_uvals = {fn : [] for fn in objective_functions}
	all_fstar = {fn : [] for fn in objective_functions}
	all_prob_success_vals = {fn : [] for fn in objective_functions}


	for trial in range(nTrials):

		indiv = np.random.uniform(0,sizeRegion,(nIndiv, 2))
		fac = np.random.uniform(0,sizeRegion,(nFac, 2))
		dist = np.array([[ (indiv[i,0] - fac[j,0])**2 + (indiv[i,1] - fac[j,1])**2 for j in range(0,nFac)] for i in range(0,nIndiv)]) # nIndiv X nFac

		yvals = {fn : np.zeros((nFac),dtype=int) for fn in objective_functions}
		rvals = {fn : np.zeros((nFac),dtype=int) for fn in objective_functions}
		uvals = {fn : np.full((nIndiv),-np.inf) for fn in objective_functions}
		fstar = {fn : 0 for fn in objective_functions}
		prob_success_vals = {fn : 0 for fn in objective_functions}

		for ind, gp in enumerate(bbax.bax_gen(nFac,nSelectedFac)):
			gp = gp.toList()
			# print(locals())
			# print(f"gp: {gp}, gp_dists: {[dist[i][j] for j in gp]}, ")
			xlist = [ gp[np.argmin([dist[i][j] for j in gp])] for i in range(nIndiv)]
			r = [dist[i][xlist[i]] for i in range(nIndiv)]

			for f in objective_functions:
				u = f(r)
				# u = [-np.log(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*r[i])) for i in range(nIndiv)]
				if sum(u) > sum(uvals[f]):
					uvals[f] = u
					rvals[f] = r
					yvals[f] = np.zeros((nFac),dtype=int)
					yvals[f][gp] = 1


		for f in objective_functions:
			fstar[f] = sum(uvals[f])
			prob_success_vals[f] = prob_success(rvals[f],beta,theta)

			all_yvals[f].extend(yvals[f])
			all_rvals[f].extend(rvals[f])
			all_uvals[f].extend(uvals[f])
			all_fstar[f].append(fstar[f])
			all_prob_success_vals[f].extend(prob_success_vals[f])
		print(f"Solved {trial+1}/{nTrials} enumeratively")

	for f in objective_functions:
		all_yvals[f] = tuple(all_yvals[f]) # so they are hashable later for grouping plots
	return all_yvals,all_rvals,all_uvals,all_fstar,all_prob_success_vals


def get_n_colors(number_desired_colors):
	# https://matplotlib.org/tutorials/colors/colormaps.ht
	cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
	colors = [cmap(i) for i in range(number_desired_colors)]
	return colors
def empiricalCDF(container):
		nElts = len(container)
		return np.array([sorted(container), [i/nElts for i in range(1,nElts + 1)]])

def plotMultiplCDFs(want_to_plot,quantityLabel):
	yvals_choices = set((v for k,v in yvals.items()))
	lumped_functions =  {choice : [fn for fn in objective_functions if yvals[fn]==choice] for choice in yvals_choices}
	lumped_labels =  {choice : ",\n".join([f"{fn.__name__} $typeLabel" for fn in objective_functions if yvals[fn]==choice]) for choice in yvals_choices}
	legend_scores = {choice : sum([fairness_strengths[fn] for fn in fns]) for choice,fns in lumped_functions.items()}
	# lumped_vals = {lumped_labels[yvals[fn]] : vals for fn,vals in want_to_plot.items()} #if yvals are the same then any want_to_plot will be the same for a subset of objectives
	lumped_vals = {yvals[fn] : vals for fn,vals in want_to_plot.items()} #if yvals are the same then any want_to_plot will be the same for a subset of objectives

	linestyles = dict(zip(set(theta),["-","--",":","-."]))
	nLines = len(lumped_functions)
	colors = get_n_colors(nLines)
	lumped_vals_separated = []
	for choice,vals in lumped_vals.items():
		c = colors.pop()
		label = lumped_labels[choice]
		for theta_val in set(theta):
			newlabel = label.replace("$typeLabel",f"type {theta_val}")
			line_content = {}
			line_content['data'] = [val for i,val in enumerate(vals) if (theta[i % nIndiv] == theta_val)]
			line_content['linestyle'] = linestyles[theta_val]
			line_content['color'] = c
			line_content['legend_score'] = legend_scores[choice] + theta_val #higher theta comes later
			lumped_vals_separated.append((newlabel,line_content))

			# lumped_vals_separated[label.replace("$typeLabel",f"type {theta_val}")] = {"color" : c, "linestyle": linestyles[theta_val], "data" : [val for i,val in enumerate(vals) if (theta[i] == theta_val)]}
	# lumped_vals_separated = {(label.replace("$typeLabel",f"type {theta_val}"),line_type) : [val for i,val in enumerate(vals) if (theta[i] == theta_val)] for label,vals in lumped_vals.items() for theta_val,line_type in zip(set(theta),linestyles) } #keys are (lineLabel,lineType,lineColor)

	lumped_vals_separated.sort(key = lambda pair: pair[1]['legend_score'])

	fig, ax = plt.subplots(figsize=(10,10))
	plt.title(f"{quantityLabel}, Empirical CDF\n(nIndividuals={nIndiv},nFacilities={nSelectedFac}/{nFac},nTrials={nTrials})")
	plt.xlabel(quantityLabel)
	plt.ylabel(f"Fraction of Individuals")

	cdfLines = [plt.plot(*empiricalCDF(line_content['data']),c=line_content['color'],label=plotLabel,ls=line_content['linestyle'],linewidth=0.5) for (plotLabel, line_content) in lumped_vals_separated]

	plt.legend(loc="best")
	plt.show(block=False)
	# plt.close(fig)
	return fig,ax








def generate_file_label():
	weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_label = f"at_{timestamp}_weights_{weights_label}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nSelectedFac}of{nFac}_nTrials_{nTrials}"
	return trial_label

trial_label = generate_file_label()


yvals,rvals,uvals,fstar,prob_success_vals = solve_multiple_enumerative()


print(trial_label)
fig,ax = plotMultiplCDFs(rvals,"Distances from Nearest Facility")
plt.savefig(f"figures/distance_cdfs_{trial_label}.pdf", bbox_inches='tight')
fig2,ax2 = plotMultiplCDFs(prob_success_vals,"Probability of Success")
plt.savefig(f"figures/prob_cdfs_{trial_label}.pdf", bbox_inches='tight')
# fig,ax = plot_region("(enumerative method)")
# fig,ax = plot_multiple_objectives("(multiple objectives)")
