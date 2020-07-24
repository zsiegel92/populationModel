import numpy as np

def sum_log_prob_weighted_factory(weight,beta,theta):
	nSamples = len(theta)
	theta_high = max(theta)
	llambda = [weight if (theta[i] != theta_high) else 1 for i in range(nSamples)]
	def sum_log_prob_weighted_function(rr):
		uu = [-llambda[i]*np.log(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]
		return uu

	sum_log_prob_weighted_function.__name__ = f"logprob_weight{weight}"
	return sum_log_prob_weighted_function

def sum_prob_weighted_factory(weight,beta,theta):
	theta_high = max(theta)
	nSamples = len(theta)
	llambda = [weight if (theta[i] != theta_high) else 1 for i in range(nSamples)]
	def sum_prob_weighted_function(rr):
		uu = [llambda[i]/(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]
		return uu
	sum_prob_weighted_function.__name__ = f"prob_weight{weight}"
	return sum_prob_weighted_function


def sum_distance_weighted_factory(weight,theta):
	theta_high = max(theta)
	nSamples = len(theta)
	llambda = [weight if (theta[i] != theta_high) else 1 for i in range(nSamples)]
	def sum_distance_weighted_function(rr):
		uu = [-llambda[i]*rr[i] for i in range(nSamples)]
		return uu
	sum_distance_weighted_function.__name__ = f"distance_weight{weight}"
	return sum_distance_weighted_function



def sum_distance(rr):
	uu = [-1*dist for dist in rr]
	return uu


def prob_success(rr,beta,theta):
	nSamples = len(theta)
	return [1/(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]

def log_prob_success(rr,beta,theta):
	nSamples = len(theta)
	return [-np.log(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]

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