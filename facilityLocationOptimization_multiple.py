import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import pickle
from subset_helper import bax
from function_factories import sum_log_prob_weighted_factory, sum_prob_weighted_factory,sum_distance_weighted_factory,prob_success,mean_distance_function,mean_probability_function,type_covariance_function, price_of_fairness_distance_function, price_of_fairness_probability_function
matplotlib.use('TKAgg') #easier window management when not using IPython
# matplotlib.rcParams['text.usetex'] = True


nIndiv = 50
nFac = 8
nSelectedFac = 4
nTrials = 500

sizeRegion = 1


theta_low = 0
theta_high = 1
theta = [theta_low for i in range(nIndiv//2)] + [theta_high for i in range(nIndiv - (nIndiv//2))]
np.random.shuffle(theta)

beta = [0,0.5,-10] #[beta0, beta_theta >0, beta_r <0]

weights = [0,0.9,0.99] #should all be >= 1
log_prob_obj_fns = [sum_log_prob_weighted_factory(w,beta,theta) for w in weights]
prob_obj_fns = [sum_prob_weighted_factory(w,beta,theta) for w in weights]
dist_obj_fns= [sum_distance_weighted_factory(w,theta) for w in weights]
objective_functions = log_prob_obj_fns + prob_obj_fns + dist_obj_fns
SWB_dist_obj = dist_obj_fns[0]
SWB_prob_obj = prob_obj_fns[0] #sum_prob_weighted_factory(1,beta,theta)
# SWB_index = objective_functions.index(SWB_prob_obj)


indices = [
	{'name':'Mean Distance','key': 'mean_dist', 'shared':False,'fn':mean_distance_function},
	{'name':'Mean Success Prob','key': 'mean_prob','shared':False,'fn':mean_probability_function},
	{'name':'Covariance(type,success)','key': 'cov','shared':True,'fn':type_covariance_function},{'name':'Price of Fairness: Distance','key': 'pof_dist','shared':True,'fn':price_of_fairness_distance_function},
	{'name':'Price of Fairness: Probability','key': 'pof_prob','shared':True,'fn':price_of_fairness_probability_function}
	]


objective_function_names = {fn.__name__ : fn for fn in objective_functions}

fairness_ratings = [10**10 + weight for weight in weights]+ [10**5 + weight for weight in weights] + weights #arbitrary, but roughly increasing with fairness-inducing tendency
fairness_strengths = dict(zip(objective_functions,fairness_ratings)) # to create a consistent order in plot legends

def solve_multiple_enumerative(saving=False):
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

	dat = {"yvals": all_yvals,"rvals": all_rvals,"uvals": all_uvals,"fstar": all_fstar,"prob_success_vals": all_prob_success_vals}

	if saving:
		save_data(dat)
	return dat
	# return all_yvals,all_rvals,all_uvals,all_fstar,all_prob_success_vals




def post_process(data, want_to_plot_key,quantityLabel):
	yvals = data['yvals']
	yvals_choices = set((v for k,v in yvals.items()))

	# SWB_dist_obj = sum_distance_weighted_factory(1,beta,theta)
	# SWB_prob_obj = sum_prob_weighted_factory(1,beta,theta)
	base_vals_dist = {k : values_dict[SWB_dist_obj] for k,values_dict in data.items()}
	base_vals_prob = {k : values_dict[SWB_prob_obj] for k,values_dict in data.items()}
	# base_vals_prob = {k : values_dict[SWB_prob_obj] for k,values_dict in data.items()}
	SWB_dist = -1*sum(base_vals_dist['rvals'])
	SWB_prob = sum(base_vals_prob['prob_success_vals']) # SWB_prob

	lumped_functions =  {choice : [fn for fn in objective_functions if yvals[fn]==choice] for choice in yvals_choices}
	lumped_labels =  {choice : ",\n".join([f"{fn.__name__} $typeLabel" for fn in objective_functions if yvals[fn]==choice]) for choice in yvals_choices}
	legend_scores = {choice : sum([fairness_strengths[fn] for fn in fns]) for choice,fns in lumped_functions.items()}
	# lumped_vals = {lumped_labels[yvals[fn]] : vals for fn,vals in want_to_plot.items()} #if yvals are the same then any want_to_plot will be the same for a subset of objectives
	want_to_plot = data[want_to_plot_key]
	#if yvals are the same then any want_to_plot will be the same for a subset of objectives
	lumped_vals = {yvals[fn] : vals for fn,vals in want_to_plot.items()}
	## for fairness
	lumped_raw_distances = {yvals[fn] : vals for fn,vals in data['rvals'].items()}
	lumped_probabilities = {yvals[fn] : vals for fn,vals in data['prob_success_vals'].items()}
	linestyles = dict(zip(set(theta),["-",":","-.","--"]))
	nLines = len(lumped_functions)
	colors = get_n_colors(nLines)
	lumped_vals_separated = []
	for choice,vals in lumped_vals.items():
		c = colors.pop()
		label = lumped_labels[choice]
		raw_distances = lumped_raw_distances[choice]
		raw_probabilities = lumped_probabilities[choice]
		raw_distances_grouped = {}
		raw_probabilities_grouped = {}

		for theta_val in set(theta):
			newlabel = label.replace("$typeLabel",f"$\\theta_{theta_val}$")
			line_content = {}
			line_content['data'] = [val for i,val in enumerate(vals) if (theta[i % nIndiv] == theta_val)]
			line_content['linestyle'] = linestyles[theta_val]
			line_content['color'] = c
			line_content['legend_score'] = legend_scores[choice] + theta_val #higher theta comes later
			raw_distances_grouped[theta_val] = [val for i,val in enumerate(raw_distances) if (theta[i % nIndiv] == theta_val)]
			raw_probabilities_grouped[theta_val] = [val for i,val in enumerate(raw_probabilities) if (theta[i % nIndiv] == theta_val)]
			line_content['raw_distances_grouped'] = raw_distances_grouped #shared reference for all theta_val
			line_content['raw_probabilities_grouped'] = raw_probabilities_grouped #shared reference for all theta_val
			line_content['theta'] = theta_val
			line_content['SWB_dist'] = SWB_dist
			line_content['SWB_prob'] = SWB_prob
			lumped_vals_separated.append((newlabel,line_content))



	for i,(label,line_content) in  enumerate(lumped_vals_separated):
		theta_val = line_content['theta']

		for index_dict in indices:
			index_name = index_dict['key']
			index_fn = index_dict['fn']
			line_content[index_name] = index_fn(line_content)
			# if index_dict['shared']:
			# 	if i % 2 == 1:
			# 		line_content[index_name] = None

		# line_content['mean_dist'] = np.mean(line_content["raw_distances_grouped"][theta_val])
		# line_content['mean_prob'] = np.mean(line_content["raw_probabilities_grouped"][theta_val])

		# indices = [{'name':'Mean Distance','key': 'mean_dist', 'shared':False,'fn':dummy},{'name':'Mean Success Prob','key': 'mean_prob','shared':False,'fn':dummy},{'name':'McCloone','key': 'mcCloone','shared':True,'fn':dummy}]

	# lumped_vals_separated.sort(key = lambda pair: pair[1]['legend_score'])
	lumped_vals_separated.sort(key = lambda pair: pair[1]['cov'],reverse=True)
	# lumped_vals_separated = {('prob_weight5 $\\theta_0$,\ndistance_weight5 $\\theta_0$',line_content),...}
	# line_content =
	return lumped_vals_separated

def get_n_colors(number_desired_colors):
	if number_desired_colors < 10:
		default = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		colors = default[0:number_desired_colors]
	else:
		# https://matplotlib.org/tutorials/colors/colormaps.ht
		cmap = plt.cm.get_cmap('nipy_spectral',number_desired_colors)
		colors = [cmap(i) for i in range(number_desired_colors)]
	return colors

def empiricalCDF(container):
		nElts = len(container)
		return np.array([sorted(container), [i/nElts for i in range(1,nElts + 1)]])



def plotMultiplCDFs(data,want_to_plot_key,quantityLabel,logPlotX=False,logPlotY=False,saving=False):
	lumped_vals_separated = post_process(data,want_to_plot_key,quantityLabel)

	# fig, (ax1,ax2) = plt.subplots(figsize=(10,10),gridspec_kw={"nrows":2,"ncols":1,"height_ratios":[2,1]})
	gs = gridspec.GridSpec(nrows=2,ncols=1,height_ratios=[3,1])
	fig = plt.figure(figsize=(15,10))
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	fig.subplots_adjust(hspace=0.5)
	decimal_formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
	if logPlotX:
		ax1.set_xscale('log') #,basex=2
		ax1.xaxis.set_major_formatter(decimal_formatter)
	if logPlotY:
		ax1.set_yscale('log')
		ax1.yaxis.set_major_formatter(decimal_formatter)

	ax1.set_title(f"{quantityLabel}, Empirical CDF\n(nIndividuals={nIndiv},nFacilities={nSelectedFac}/{nFac},nTrials={nTrials})")
	log_note = (""," (log scale)") # can be indexed by True/False as 0/1
	ax1.set_xlabel(quantityLabel + log_note[logPlotX])
	ax1.set_ylabel(f"Fraction of Individuals" + log_note[logPlotY])
	cdfLines = [ax1.plot(*empiricalCDF(line_content['data']),c=line_content['color'],label=plotLabel,ls=line_content['linestyle'],linewidth=1,alpha=0.5) for (plotLabel, line_content) in lumped_vals_separated]

	leg = ax1.legend(loc="best",prop={"size":8})
	# set the linewidth of each legend object
	for legobj in leg.legendHandles:
	    legobj.set_linewidth(3.0)



	nRows = len(lumped_vals_separated)
	nCols = len(indices)
	cols = [index['name'] for index in indices]
	colors = [line_content['color'] for (plotLabel, line_content) in lumped_vals_separated]
	rows = [plotLabel for (plotLabel, line_content) in lumped_vals_separated]
	content = [["{:.8f}".format(line_content[index_dict['key']]) for index_dict in indices] for line_name,line_content in lumped_vals_separated]

	# content.reverse()
	# colors = get_n_colors(nRows)
	# Adjust layout to make room for the table:
	# plt.subplots_adjust(bottom=0.3)


	ax2.axis('off')
	table = ax2.table(cellText = content,
					  rowColours=colors,
					  rowLabels=rows,
					  colLabels=cols,
					  loc='center',
					  cellLoc='left')
	table_d = table.get_celld()

	table.set_fontsize(8)

	# # shared cells
	for i in range(nRows):
		row_ind = i + 1
		for j in range(0,nCols):
			if indices[j]['shared']:
				if row_ind % 2 == 1:
					table_d[(row_ind,j)].visible_edges = 'RTL'
					table_d[(row_ind,j)].set_text_props(verticalalignment='bottom')
				else:
					table_d[(row_ind,j)].visible_edges = 'BRL'
					table_d[(row_ind,j)].set_text_props(text='')

	#set row heights
	for i in range(1,nRows+1):
		base_h = 0.11
		hh = base_h*(rows[i-1].count('\n')+1)
		for j in range(-1,nCols):
			table_d[(i,j)].set_height(hh)

	plt.show(block=False)
	# plt.close(fig)
	log_identifier = ''
	if logPlotX:
		log_identifier += 'logX_'
	if logPlotY:
		log_identifier += 'logY_'
	if saving:
		plt.savefig(f"figures/{want_to_plot_key}_cdfs_{log_identifier}{generate_file_label()}.pdf", bbox_inches='tight')
	return fig,(ax1,ax2)
	# return table


def generate_file_label():
	weights_label = "_".join(map(str,weights))
	beta_label = "_".join(map(str,beta))
	timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
	trial_label = f"at_{timestamp}_weights_{weights_label}_beta_{beta_label}_nIndiv_{nIndiv}_nFac_{nSelectedFac}of{nFac}_nTrials_{nTrials}"
	return trial_label

def save_data(dat):
	# dat = {"yvals" : yvals,"rvals" : rvals,"uvals" : uvals,"fstar" : fstar,"prob_success_vals" : prob_success_vals}
	named_dat = {name: {fn.__name__ : val for fn,val in fndict.items()} for name,fndict in dat.items()}
	data_filename = f"data/data_{generate_file_label()}.pickle"
	pickle.dump(named_dat,open( data_filename, "wb" ))

def load_data(data_filename):
	dat = pickle.load(open(data_filename,"rb" ))
	dat = {name: {objective_function_names[fn_name] : val for fn_name,val in fn_name_dict.items()} for name,fn_name_dict in dat.items()}
	return dat



trial_label = generate_file_label()

saving = True
dat = solve_multiple_enumerative(saving=saving)

# dat = load_data('data/data_at_23_07_2020_21_07_weights_1_5_1000_beta_0_0.2_-1_nIndiv_50_nFac_4of8_nTrials_40.pickle')


plotMultiplCDFs(dat,'rvals',"Distances from Nearest Facility",logPlotX=False,saving=saving)

# plotMultiplCDFs(dat,'rvals',"Distances from Nearest Facility (log scale x)",logPlotX=True)


plotMultiplCDFs(dat,'rvals',"Distances from Nearest Facility",logPlotX=True,logPlotY=True,saving=saving)

# plotMultiplCDFs(dat,'prob_success_vals',"Probability of Success",logPlotX=False)
# plotMultiplCDFs(dat,'prob_success_vals',"Probability of Success (log scale x)",logPlotX=True)
plotMultiplCDFs(dat,'prob_success_vals',"Probability of Success",logPlotY=True,saving=saving)

# plotMultiplCDFs(dat,'prob_success_vals',"Probability of Success (log log)",logPlotX=True,logPlotY=True)
