''' Estimate the most likely labels using task outputs from multiple workers. Two approaches are supported: EM algorithm and majority voting.
The EM algorithm is described in "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm (Dawid and Skene, 1979)"
'''

import numpy as np
import pandas as pd
import math
import itertools
import time


def rand_observations(objects, workers, labels):
	''' Generate random observations
	Args:
		objects: Set of objects
		workers: Set of workers
		labels: Set of labels
	Return:
		DataFrame of random observations where rows representing objects and columns representing workers
	'''
	return pd.DataFrame(np.random.randint(len(labels), size=(len(objects), len(workers))), index=objects, columns=workers)


def generate_true_estimates(true_estimates, labels):
	''' Generate true estimates DataFrame
	Args:
		true_estimates: List of true estimates as tuples (object index, label index)
		labels: Set of label indices
	'''
	x = []
	idx = []
	for o, l in true_estimates:
		v = np.zeros(len(labels))
		v[l] = 1
		x.append(v)
		idx.append(o)
	
	return pd.DataFrame(x, index=idx, columns=labels)


class LabelEstimator(object):
	''' Estimate true labels from a set of observations
		Outputs of the EM algorithm: Class priors, error rates, and estimated labels, are accessible via class attributes, respectively: em_p, em_pi, em_T
	'''

	def __init__(self, observations, objects, workers, labels, true_estimates=None):
		''' Instantiate the estimator with specific parameters
		Args:
			observations: Observations where each tuple comprising (object index, worker index, label index)
			true_estimates: Known true estimates
			objects: Set of indices representing objects
			workers: Set of indices representing workers
			labels: Set of indices representing labels
		'''
		self.observations = observations
		self.objects = objects
		self.workers = workers
		self.labels = labels
		if true_estimates is not None:
			self.true_estimates = true_estimates
		else:
			self.true_estimates = None

		# Variables for storing clas priors (p), error rates (pi), and estimated labels (T) at the end of EM steps 
		self.em_p = None
		self.em_pi = None
		self.em_T = None

		# Generate n^k_il: k = workers, i = objects, l = labels
		self.n = self._worker_object_responses()


	def _worker_object_responses(self):
		''' Generate n^k_il assuming that each worker works with an object only once
		Return:
			Series representing responses from workers on corresponding objects
		'''
		tuples = []
		for i in self.objects:
			for k in self.workers:
				tuples.append((k, i, self.observations.loc[i, k]))

		n = pd.Series(index=pd.MultiIndex.from_tuples(tuples, names=['k', 'i', 'l']))
		n = n.fillna(1)
		return n


	def _init_estimates(self):
		''' Initialize T_ij = \sum_k{n^k_il}/sum_k{sum_k{n^k_il}} (eq. 3.1 in Dawid and Skene 1979)
			This is the same as majority voting estimates
		Return:
			DataFrame of initial object-label estimates
		'''
		T_ij = pd.DataFrame(index=self.objects, columns=self.labels)
		for i in T_ij.index:
			for j in T_ij.columns:
				T_ij.loc[i, j] = round(float(len(self.observations.loc[i][self.observations.loc[i]==j]))/len(self.observations.columns), 4)

		return T_ij


	def _marginal_probabilities(self, T_ij):
		''' Compute marginal probabilities p_j = \sum_i{T_ij}/|I| (eq. 2.4 in Dawid and Skene 1979)
		Args:
			T_ij: Object-label estimates
		Return:
			Array of marginal probabilities
		'''
		p_j = []
		for j in T_ij.columns:
			p_j.append(T_ij[j].sum()/len(T_ij.index))
		return p_j


	def _error_rates(self, T_ij):
		''' Compute error-rates pi^k_jl = \sum_j{T_ij*n^k_il}/sum_l{sum_i{T_ij*n^k_il}} (eq. 2.3 in Dawid and Skene 1979)
		Args:
			T_ij: Estimates
		Return:
			Series representing error-rates
		'''
		tuples = [x for x in itertools.product(*[self.workers, self.labels, self.labels])]
		pi = pd.Series(index=pd.MultiIndex.from_tuples(tuples, names=['k', 'j', 'l']))
		pi_nom = pd.Series(index=pd.MultiIndex.from_tuples(tuples, names=['k', 'j', 'l']))
		tuples = [x for x in itertools.product(*[self.workers, self.labels])]
		pi_denom = pd.Series(index=pd.MultiIndex.from_tuples(tuples, names=['k', 'j']))
		
		for k in set(self.n.index.get_level_values('k')):
			for j in T_ij.columns:
				denom = 0
				for l in set(self.n.index.get_level_values('l')):
					nom = 0
					for i in set(self.n.index.get_level_values('i')):
						if (k, i, l) in self.n.index:
							nom += (T_ij.loc[i, j] * self.n.loc[k, i, l])
					pi_nom.loc[k, j, l] = nom
					denom += nom
				pi_denom.loc[k, j] = denom

		for k, j, l in pi.index:
			pi.loc[k, j, l] = pi_nom.loc[k, j, l]/pi_denom.loc[k, j]
		return pi


	def _em_estimator(self, T_hat=None, iteration=None, prev_loglik=None, threshold=0.001, max_iteration=50):
		''' Predict labels using EM algorithm
		Args:
			T_hat: Starting estimates
			iteration: Current iteration
			prev_loglik: log-likelihood of the previous iteration
			threshold: log-likelihood threshold as a termination criteria
			max_iteration: Maximum number of iterations
		Return:
			T: Object-label estimates
		'''
		t0 = time.time()
		# Step 1: Initialize T_ij = \sum_k{n^k_il}/sum_k{sum_k{n^k_il}}
		if T_hat is None:
			T_hat = self._init_estimates()

		if self.true_estimates is not None:
			for i in self.true_estimates.index:
				T_hat.loc[i] = self.true_estimates.loc[i]

		if iteration is None:
			iteration = 0
		else:
			iteration += 1

		# Step 2: Compute marginal probability p_j and error-rates pi^k_jl
		p = self._marginal_probabilities(T_hat)
		pi = self._error_rates(T_hat)

				
		# Step 3: Re-compute T_ij (eq. 2.5 in Dawid and Skenen 1979)
		# p(T_ij=1 | data) = (p_j * \product_k{\product_l{pi^k_jl **  n^K_il}}) / (\sum_q{p_q * \product_k{\product_l{pi^k_ql **  n^K_il}}})
		T = pd.DataFrame(index=self.objects, columns=self.labels)
		tuples = [x for x in itertools.product(*[T.index, T.columns])]
		T_nom = pd.Series(index=pd.MultiIndex.from_tuples(tuples, names=['i', 'j']))
		T_denom = []

		# Compute log-likelihood for full data (eq 2.7 in Dawid and Skene 1979)
		x1 = 0
		x2 = 0
		loglik = 0

		for i in T.index:
			denom = 0
			for j in T.columns:
				nom = 1
				for k in set(pi.index.get_level_values('k')):
					for l in set(pi.index.get_level_values('l')):
						if (k, i, l) in self.n.index:
							nom *= math.pow(pi.loc[k, j, l], self.n.loc[k, i, l])
							if pi.loc[k, j, l] > 0: 
								x1 += self.n.loc[k, i, l] * math.log(pi.loc[k, j, l])
				nom *= p[j]
				x2 += x1 * p[j]
				T_nom.loc[i, j] = nom
				denom += nom
			loglik += x2
			T_denom.append(denom)


		for i, j in T_nom.index:
			l = T_nom.loc[i, j]/T_denom[i]
			T.loc[i, j] = round(l, 4)

		# Storing outputs
		self.em_p = p
		self.em_pi = pi
		self.em_T = T

		if iteration == 0:
			return self._em_estimator(T, iteration, loglik, threshold, max_iteration)
		elif iteration >= max_iteration:
			return T

		t1 = time.time()

		# Step 4: If converge, terminate; otherwise repeate 2 and 3
		diff = round(math.fabs(loglik-prev_loglik)/math.fabs(loglik), 4)
		if diff > threshold:
			print 'iteration %s:/%s loglik=%s,  prev_loglik=%s, diff_pct=%.4f, time=%.4f sec' % (iteration, max_iteration, loglik, prev_loglik, diff, (t1-t0))
			return self._em_estimator(T, iteration, loglik, threshold, max_iteration)
		
		print 'iteration %s/%s: loglik=%s,  prev_loglik=%s, diff_pct=%.4f, time=%.4f sec' % (iteration, max_iteration, loglik, prev_loglik, diff, (t1-t0))
		return T 


	def _mv_estimator(self):
		''' Predict levels using majority voting T_ij = \sum_k{n^k_il}/sum_k{sum_k{n^k_il}}
		Return:
			DataFrame representing object-label estimates
		'''
		T = pd.DataFrame(index=self.objects, columns=self.labels)
		for i in T.index:
			for j in T.columns:
				T.loc[i, j] = round(float(len(self.observations.loc[i][self.observations.loc[i]==j]))/len(self.observations.columns), 4)

		if self.true_estimates is not None:
			for i in self.true_estimates.index:
				T.loc[i] = self.true_estimates.loc[i]

		return T


	def _em_worker_cost(self):
		''' Estimate the expected cost of each worker (algorithm 2 in Ipeirotis et al. 2010)
			Perfect workers will have zero cost while random workers/spammers will have high cost
		'''
		cost = []

		# for each worker k, estimate worker's probability of assigning labels using eq. 2
		X = pd.DataFrame(index=self.workers, columns=self.labels)
		X = X.fillna(0)

		for k in X.index:
			for l in X.columns:
				for j in self.labels:
					X.loc[k, l] += self.em_pi.loc[k, j, l] * self.em_p[j]

		print 'Pr(AC=l):\n %s' % X

		for k in self.workers:
			c = 0
			for l in self.labels:
				# Compute soft label using eq. 1
				soft = []
				for j in self.labels:
					soft.append((self.em_pi.loc[k, j, l] * self.em_p[j])/X.loc[k, l])
				print 'k=%s, l=%s, soft=%s' % (k, l, soft)

				# Compute cost c using eq. 3. Use simple c_ij where c_ij = 0 iff i == j; 1 iff i != j
				c_soft = 0
				for i in self.labels: 
					for j in self.labels:
						if i == j:
							c_ij = 0
						else:
							c_ij = 1
						if not math.isnan(soft[i]) and not math.isnan(soft[j]):
							c_soft += soft[i] * soft[j] * c_ij
				
				c += c_soft * X.loc[k, l]

			cost.append(c) 

		print 'Cost: %s' % cost
		return cost


	def predict(self, method, max_iteration=None):
		''' Predict labels using EM algorithm or majority voting
		Args:
			method: "em" for EM algorithm, "mv" for majority voting
			max_iteration: Maximum number of iterations for EM algorithm
		Returns:
			DataFrame where rows representing objects, columns representing labels, and cells representing probabiliites
		'''
		if method=='em':
			if max_iteration is not None:
				estimators = self._em_estimator(None, max_iteration=max_iteration)
			else:
				estimators = self._em_estimator(None)

			return estimators
		
		elif method=='mv':
			return self._mv_estimator()
		else:
			raise Exception('Error: Unrecognized method "'+method+'"')


if __name__ == '__main__':
	objects = np.arange(5)
	workers = np.arange(5)
	labels = np.arange(2)
	observations = rand_observations(objects, workers, labels)
	print 'Observation:\n%s' % observations
	true_estimates = generate_true_estimates([(1,0), (3,0), (5,0)], labels)

	est = LabelEstimator(observations, objects, workers, labels, true_estimates)
	t0 = time.time()
	T = est.predict('em')
	t1 = time.time()
	print 'Done in %.4f sec' % (t1-t0)
	print 'EM:\n%s' % T
	# print 'Class priors: %s' % est.em_p
	# print 'Error rates: %s' % est.em_pi
	# T = est.predict('mv')
	# print 'Majority voting:\n%s' % T

	est._em_worker_cost()