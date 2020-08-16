import numpy as np
mpl_markers = ['.','v','P','*','+','x','_']
markers = ['\\times','\\bigcirc','+','\\Box','\\dagger','\\Diamond','\\oplus','\\wedge','\\oslash','\\bullet','\\odot','\\triangleright','\\otimes',]


def create_tex_marker_with_number(marker,number):
	# weight_ind = weights.index(number)
	# ind = markers.index(marker)
	# if ind == 0:
	# 	return f"${marker}^{{{number}}}$"
	# elif ind == 1:
	# 	return f"${marker}_{{{number}}}$"
	# elif ind == 2:
	# 	return f"${{}}^{{{number}}}\\!{marker}$"
	# elif ind == 3:
	# 	return f"${{}}_{{{number}}}\\!{marker}$"
	# else:
	# 	return f"${marker}{number}$"
	return f"${marker}^{{{number}}}$"

def sum_log_prob_weighted_factory(weight):
	high_weight = (1-weight)/2
	low_weight = (1+weight)/2
	def sum_log_prob_weighted_function(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		llambda = [low_weight if instance.theta[i] != instance.theta_high else high_weight for i in range(nSamples) ]
		uu = np.array([-llambda[i]*np.log(1+np.exp(-instance.beta[0] - instance.beta[1]*instance.theta[i] - instance.beta[2]*rr[i])) for i in range(nSamples)])
		return uu
	the_fn = sum_log_prob_weighted_function
	the_fn.must_abs = False
	the_fn.coeff = "\\alpha"
	the_fn.basebasename = f"logprob"
	the_fn.basename = f"{the_fn.basebasename}_weight${the_fn.coeff}$"
	the_fn.__name__ = f"logprob_weight{weight}"
	the_fn.weight = weight
	the_fn.weightstr = str(the_fn.weight)
	the_fn.marker = markers[0]
	the_fn.mpl_marker = mpl_markers[0]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight}}}$"
	the_fn.basemarker = f"${the_fn.marker}^{{{the_fn.coeff}}}$"
	return the_fn

def sum_prob_weighted_factory(weight):
	high_weight = (1-weight)/2
	low_weight = (1+weight)/2
	def sum_prob_weighted_function(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		llambda = [low_weight if instance.theta[i] != instance.theta_high else high_weight for i in range(nSamples) ]
		uu =np.array( [llambda[i]/(1+np.exp(-instance.beta[0] - instance.beta[1]*instance.theta[i] - instance.beta[2]*rr[i])) for i in range(nSamples)])
		return uu
	the_fn = sum_prob_weighted_function
	the_fn.must_abs = False
	the_fn.coeff = "\\alpha"
	the_fn.basebasename = f"prob"
	the_fn.basename = f"{the_fn.basebasename}_weight${the_fn.coeff}$"
	the_fn.__name__ = f"prob_weight{weight}"
	the_fn.weight = weight
	the_fn.weightstr = str(the_fn.weight)
	the_fn.marker = markers[1]
	the_fn.mpl_marker = mpl_markers[1]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight}}}$"
	the_fn.basemarker = f"${the_fn.marker}^{{{the_fn.coeff}}}$"
	return the_fn


def sum_distance_weighted_factory(weight):
	high_weight = (1-weight)/2
	low_weight = (1+weight)/2
	def sum_distance_weighted_function(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		llambda = [low_weight if instance.theta[i] != instance.theta_high else high_weight for i in range(nSamples) ]
		# llambda = np.ones(nSamples)
		# llambda[instance.theta_low_indices] = weight
		# uu = np.array([-llambda[i]*rr[i] for i in range(nSamples)])
		uu = np.array([-llambda[i]*rr[i] for i in range(nSamples)])
		return uu
	the_fn = sum_distance_weighted_function
	the_fn.must_abs = False
	the_fn.coeff = "\\alpha"
	the_fn.basebasename = f"distance"
	the_fn.basename = f"{the_fn.basebasename}_weight${the_fn.coeff}$"
	the_fn.__name__ = f"distance_weight{weight}"
	the_fn.weight = weight
	the_fn.weightstr = str(the_fn.weight)
	the_fn.marker = markers[2]
	the_fn.mpl_marker = mpl_markers[2]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight}}}$"
	the_fn.basemarker = f"${the_fn.marker}^{{{the_fn.coeff}}}$"
	return the_fn


# Returns negative of covariance - maximize to minimize covariance
def covariance_factory():
	def covariance(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		avg_theta = (1/nSamples) * sum(instance.theta) #should be 0.5
		uu = np.array( [-(1/nSamples)* (instance.theta[i] - avg_theta) / (1+np.exp(-instance.beta[0] - instance.beta[1]*instance.theta[i] - instance.beta[2]*rr[i])) for i in range(nSamples)]) #make utilities negative so as to MINIMIZE covariance (encourage higher type -> LOWER utility)
		return uu
	the_fn = covariance
	the_fn.must_abs = True

	# def summer(instance):
	# 	return abs(sum(the_fn(instance)))
	# the_fn.summer = summer
	the_fn.coeff = ""
	the_fn.basename = f"Cov(type,P(success))"
	the_fn.__name__ = f"Cov(type,P(success))"
	the_fn.marker = markers[4]
	the_fn.weight = ""
	the_fn.weightstr = str(the_fn.weight)
	the_fn.mpl_marker = mpl_markers[4]
	the_fn.weighted_marker = f"${the_fn.marker}$"
	the_fn.basemarker = f"${the_fn.marker}^{{{the_fn.coeff}}}$"
	return the_fn

def linear_combination_of_objectives(fn1,fn2,weight_fn2):
	fn1_weight = (1-weight_fn2)/2
	fn2_weight = (1+weight_fn2)/2
	def linear_comb(instance):
		return fn1_weight * fn1(instance) + fn2_weight * fn2(instance)
	the_fn = linear_comb
	the_fn.must_abs = True
	# def summer(vals):
	# 	if fn1.self_sum:
	# 		sum1 = fn1.summer(fn1(instance))
	# 	else:
	# 		sum1 = sum(fn1(instance))
	# 	if fn2.self_sum:
	# 		sum2 = fn2.summer(fn2(instance))
	# 	else:
	# 		sum2 = sum(fn2(instance))
	# 	return sum1 + weight_fn2 * sum2
	# the_fn.summer = summer
	the_fn.coeff = "\\gamma"
	the_fn.basename = f"{fn1.basename}+${the_fn.coeff}${fn2.basename}"
	the_fn.__name__ = f"{fn1.__name__}+{weight_fn2}{fn2.__name__}"
	# the_fn.weights = (fn1.weight,fn2.weight)
	the_fn.weight = weight_fn2
	the_fn.weight_fn2 = weight_fn2
	the_fn.weightstr = f"{weight_fn2}"#f"{fn1.weightstr}+{the_fn.weight_fn2}*{fn2.weightstr}"
	the_fn.markers = (fn1.marker,fn2.marker)
	the_fn.marker = markers[4]
	the_fn.mpl_marker = mpl_markers[4]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight_fn2}}}_{{{fn1.weight}{',' if fn2.weight else ''}{fn2.weight}}}$"
	the_fn.basemarker = f"${the_fn.marker}^{{{the_fn.coeff}}}_{{{fn1.coeff}{',' if fn2.coeff else ''}{fn2.coeff}}}$"
	return the_fn

# def sum_distance(rr):
# 	uu = [-1*dist for dist in rr]
# 	return uu


# def prob_success(rr,beta,theta):
# 	nSamples = len(theta)
# 	return [1/(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]

# def log_prob_success(rr,beta,theta):
# 	nSamples = len(theta)
# 	return [-np.log(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]

def mean_distance_function(line_content):
	theta_val = line_content['theta']
	raw_distances = line_content["raw_distances_grouped"]
	return np.mean(raw_distances[theta_val])

def mean_probability_function(line_content):
	theta_val = line_content['theta']
	raw_probabilities = line_content["raw_probabilities_grouped"]
	return np.mean(raw_probabilities[theta_val])

def type_covariance_function(line_content):
	raw_probabilities = line_content["raw_probabilities_grouped"] #of the form {0 : [probs...],1: [probs...]}
	nIndiv = sum([len(values) for theta_val,values in raw_probabilities.items()])
	theta_mean = (1/nIndiv)* sum([theta_val * len(values) for theta_val,values in raw_probabilities.items()]) #should be 0.5
	cov = (1/nIndiv) * sum([(theta_val - theta_mean) * prob for theta_val in raw_probabilities for prob in raw_probabilities[theta_val]])
	return cov


def price_of_fairness_distance_function(line_content):
	raw_distances = line_content["raw_distances_grouped"]
	all_distances = []
	for theta_val,distances in raw_distances.items():
		all_distances += distances
	SWB_dist = line_content['SWB_dist']
	SWB_fair_dist = -sum(all_distances)
	return (SWB_dist - SWB_fair_dist)/	abs(SWB_dist)


def price_of_fairness_probability_function(line_content):
	raw_probabilities = line_content["raw_probabilities_grouped"]
	all_probabilities = []
	for theta_val,probabilities in raw_probabilities.items():
		all_probabilities += probabilities
	SWB_prob = line_content['SWB_prob']
	return (SWB_prob - sum(all_probabilities))/	abs(SWB_prob)
