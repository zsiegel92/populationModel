import numpy as np

def sum_log_prob_weighted_factory(weight,beta,theta):
	nSamples = len(theta)
	theta_high = max(theta)
	def sum_log_prob_weighted_function(rr):
		theta_high = max(theta)
		llambda = [weight if (theta[i] == theta_high) else 1 for i in range(nSamples)]
		uu = [-llambda[i]*np.log(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]
		return uu

	sum_log_prob_weighted_function.__name__ = f"sum_log_prob_weighted_{weight}"
	return sum_log_prob_weighted_function

def sum_prob_weighted_factory(weight,beta,theta):
	theta_high = max(theta)
	nSamples = len(theta)
	def sum_prob_weighted_function(rr):
		llambda = [weight if (theta[i] == theta_high) else 1 for i in range(nSamples)]
		uu = [llambda[i]/(1+np.exp(-beta[0] - beta[1]*theta[i] - beta[2]*rr[i])) for i in range(nSamples)]
		return uu
	sum_prob_weighted_function.__name__ = f"sum_prob_weighted_{weight}"
	return sum_prob_weighted_function


def sum_distance_weighted_factory(weight,theta):
	theta_high = max(theta)
	nSamples = len(theta)
	def sum_distance_weighted_function(rr):
		llambda = [weight if (theta[i] == theta_high) else 1 for i in range(nSamples)]
		uu = [-llambda[i]*rr[i] for i in range(nSamples)]
		return uu
	sum_distance_weighted_function.__name__ = f"sum_distance_weighted_{weight}"
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


