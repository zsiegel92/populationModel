import numpy as np
# markers = ['.','v','P','*','+','x','_']
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
	def sum_log_prob_weighted_function(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		llambda = [weight if instance.theta[i] != instance.theta_high else 1 for i in range(nSamples) ]
		uu = np.array([-llambda[i]*np.log(1+np.exp(-instance.beta[0] - instance.beta[1]*instance.theta[i] - instance.beta[2]*rr[i])) for i in range(nSamples)])
		return uu
	the_fn = sum_log_prob_weighted_function
	the_fn.coeff = "\\lambda"
	the_fn.basename = f"logprob_weight${the_fn.coeff}$"
	the_fn.__name__ = f"logprob_weight{weight}"
	the_fn.weight = weight
	the_fn.marker = markers[0]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight}}}$"
	return the_fn

def sum_prob_weighted_factory(weight):
	def sum_prob_weighted_function(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		llambda = [weight if instance.theta[i] != instance.theta_high else 1 for i in range(nSamples) ]
		uu =np.array( [llambda[i]/(1+np.exp(-instance.beta[0] - instance.beta[1]*instance.theta[i] - instance.beta[2]*rr[i])) for i in range(nSamples)])
		return uu
	the_fn = sum_prob_weighted_function
	the_fn.coeff = "\\lambda"
	the_fn.basename = f"prob_weight${the_fn.coeff}$"
	the_fn.__name__ = f"prob_weight{weight}"
	the_fn.weight = weight
	the_fn.marker = markers[1]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight}}}$"
	return the_fn


def sum_distance_weighted_factory(weight):
	def sum_distance_weighted_function(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		llambda = [weight if instance.theta[i] != instance.theta_high else 1 for i in range(nSamples) ]
		# llambda = np.ones(nSamples)
		# llambda[instance.theta_low_indices] = weight
		# uu = np.array([-llambda[i]*rr[i] for i in range(nSamples)])
		uu = np.array([-llambda[i]*rr[i] for i in range(nSamples)])
		return uu
	the_fn = sum_distance_weighted_function
	the_fn.coeff = "\\lambda"
	the_fn.basename = f"distance_weight${the_fn.coeff}$"
	the_fn.__name__ = f"distance_weight{weight}"
	the_fn.weight = weight
	the_fn.marker = markers[2]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight}}}$"
	return the_fn


# Returns negative of covariance - maximize to minimize covariance
def covariance_factory():
	def covariance(instance):
		rr = instance.rvals
		nSamples = len(instance.rvals)
		avg_theta = (1/nSamples) * sum(instance.theta) #should be 0.5
		uu =np.array( [-(1/nSamples)* (instance.theta[i] - avg_theta) / (1+np.exp(-instance.beta[0] - instance.beta[1]*instance.theta[i] - instance.beta[2]*rr[i])) for i in range(nSamples)])
		return uu
	the_fn = covariance
	the_fn.coeff = "\\lambda"
	the_fn.basename = f"Cov(type,P(success))"
	the_fn.__name__ = f"Cov(type,P(success))"
	the_fn.marker = markers[4]
	the_fn.weighted_marker = f"${the_fn.marker}$"
	return the_fn

def linear_combination_of_objectives(fn1,fn2,weight_fn2):
	def linear_comb(instance):
		return fn1(instance) + weight_fn2*fn2(instance)
	the_fn = linear_comb
	the_fn.coeff = "\\gamma"
	the_fn.basename = f"{fn1.basename}+${the_fn.coeff}${fn2.basename}"
	the_fn.__name__ = f"{fn1.__name__}+{weight_fn2}{fn2.__name__}"
	# the_fn.weights = (fn1.weight,fn2.weight)
	the_fn.weight_fn2 = weight_fn2
	the_fn.markers = (fn1.marker,fn2.marker)
	the_fn.marker = markers[4]
	the_fn.weighted_marker = f"${the_fn.marker}^{{{the_fn.weight_fn2}}}$"
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
