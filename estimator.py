''' Estimate the most likely labels from crowdsourcing tasks. 
	Two approaches are implemented: EM algorithm and majority voting.
	The EM algorithm is described in "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm (Dawid and Skene, 1979)".
	Workers' cost derived from EM is estimated by the equations 1, 2, and 3 in "Quality Management on Amazon Mechnical Turk (Ipeirotis et al. 2010)".
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
		Observations
	'''
	return pd.DataFrame(np.random.randint(len(labels), size=(len(objects), len(workers))), index=objects, columns=workers)


def generate_true_estimates(true_estimates, labels):
	''' Generate true estimates DataFrame
	Args:
		true_estimates: List of true estimates as tuples (object index, label index)
		labels: Set of label indices
	Return:
		True label estimates
	'''
	x = []
	idx = []
	for o, l in true_estimates:
		v = np.zeros(len(labels))
		v[l] = 1
		x.append(v)
		idx.append(o)
	
	return pd.DataFrame(x, index=idx, columns=labels)


class ObservationGenerator(object):
	''' Generate observation data of objects-workers matrix
		Attributes:
			self.obs: Generated observation dataframe
			self.true_labels: True labels
			self.workers_types: Labeled workers by types: Expert, normal, random spammer, sloppy, and uniform spammer
			self.r: Reliability rating of each worker type
			self.dist: Worker type distribution
	'''

	def __init__(self, objects, workers, labels, dist):
		''' Generate observations according to specific parameters
			Args:
				objects: Set of indices representing objects
				workers: Set of indices representing workers
				labels: Set of indices representing labels
				dist: Worker type distribution, i.e. [p_expert, p_normal, p_random, p_sloppy, p_uniform]
		'''
		self.obs = pd.DataFrame(index=objects, columns=workers)
		self.true_labels = np.random.randint(max(labels)+1, size=len(objects))
		worker_types = ['expert', 'normal', 'random', 'sloppy', 'uniform']
		self.r = {'expert': (0.9, 1), 'normal': (0.6, 0.9), 'random': (0.4, 0.6), 'sloppy': (0.1, 0.4), 'uniform': None}
		if dist == None:
			self.dist = [0.2, 0.2, 0.2, 0.2, 0.2]
		if int(sum(dist)) > 1:
			raise Exception('Probabilities do not sum up to 1')
		self.dist = dist
		self.workers_types = np.random.choice(worker_types, len(workers), p=self.dist)
		
		for j in self.obs.columns:
			w = self.workers_types[j]
			if w != 'uniform':
				num_obj = len(objects) * np.random.uniform(self.r[w][0], self.r[w][1])
				correct_answer_indexes = np.random.permutation(np.arange(len(objects)))[:num_obj]
				for i in self.obs.index:
					if i in correct_answer_indexes:
						self.obs.loc[i, j] = self.true_labels[i]
					else:
						l = list(labels.copy())
						l.remove(self.true_labels[i])
						self.obs.loc[i, j] = np.random.choice(l, 1)[0]
			else:
				l = np.random.choice(labels, 1)[0]
				self.obs[j] = self.obs[j].fillna(value=l)


class LabelEstimator(object):
	''' Estimate true labels from a set of observations
		Attributes
			self.T: Estimated correct labels (rows=objects, columns=labels)
		Atrributes specific to EM algorithm: 
			self.em_p: Estimated class priors (columns=labels)
			self.em_pi: Estimated error rates (MultiIndex (k=worker, j=correct label, l=assigned label))
			self.em_C: Estimated workers' cost (columns=workers)
	'''

	def __init__(self, observations, objects, workers, labels, true_estimates=None):
		''' Instantiate the estimator with specific parameters
		Args:
			observations: Observations where each tuple comprising (object index, worker index, label index)
			true_estimates: True correct labels
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
		self.T = None
		self.em_p = None
		self.em_pi = None
		self.em_C = None

		# Generate n^k_il: k = workers, i = objects, l = labels
		self.n = self._workers_objects_labels()


	def _workers_objects_labels(self):
		''' Generate n^k_il assuming that each worker works with an object only once
		Return:
			Workers (k) ' assigned labels (l) for specific objects (i)
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
			Estimated correct labels
		'''
		T_ij = pd.DataFrame(index=self.objects, columns=self.labels)
		for i in T_ij.index:
			for j in T_ij.columns:
				T_ij.loc[i, j] = round(float(len(self.observations.loc[i][self.observations.loc[i]==j]))/len(self.observations.columns), 4)

		return T_ij


	def _class_priors(self, T_ij):
		''' Compute marginal probabilities p_j = \sum_i{T_ij}/|I| (eq. 2.4 in Dawid and Skene 1979)
		Args:
			T_ij: Estimated correct labels
		Return:
			Estimated class priors
		'''
		p_j = []
		for j in T_ij.columns:
			p_j.append(float(T_ij[j].sum())/len(T_ij.index))
		return p_j


	def _error_rates(self, T_ij):
		''' Compute error-rates pi^k_jl = \sum_j{T_ij*n^k_il}/sum_l{sum_i{T_ij*n^k_il}} (eq. 2.3 in Dawid and Skene 1979)
		Args:
			T_ij: Estimates
		Return:
			Estimated error-rates
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


	def _em_estimator(self, T_hat=None, iteration=None, prev_loglik=None, threshold=0.001, max_iteration=50, verbose=False):
		''' Estimate correct labels using EM algorithm
		Args:
			T_hat: Starting estimates
			iteration: Current iteration
			prev_loglik: log-likelihood of the previous iteration
			threshold: log-likelihood threshold as a termination criteria
			max_iteration: Maximum number of iterations
		Return:
			Estimated correct labels
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
		p = self._class_priors(T_hat)
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

		self.T = T
		self.em_p = p
		self.em_pi = pi

		if iteration == 0:
			return self._em_estimator(T, iteration, loglik, threshold, max_iteration, verbose)
		elif iteration >= max_iteration:
			return T

		t1 = time.time()

		# Step 4: If converge, terminate; otherwise repeate 2 and 3
		diff = round(math.fabs(loglik-prev_loglik)/math.fabs(loglik), 4)
		if diff > threshold:
			if verbose:
				print 'iteration %s:/%s loglik=%.4f,  prev_loglik=%.4f, diff_pct=%.4f, time=%.4f sec' % (iteration, max_iteration, loglik, prev_loglik, diff, (t1-t0))
			return self._em_estimator(T, iteration, loglik, threshold, max_iteration, verbose)
		
		if verbose:
			print 'iteration %s/%s: loglik=%.4f,  prev_loglik=%.4f, diff_pct=%.4f, time=%.4f sec' % (iteration, max_iteration, loglik, prev_loglik, diff, (t1-t0))
		return T 


	def _mv_estimator(self):
		''' Estimate correct labels using majority voting T_ij = \sum_k{n^k_il}/sum_k{sum_k{n^k_il}}
		Return:
			Estimated correct labels
		'''
		T = pd.DataFrame(index=self.objects, columns=self.labels)
		for i in T.index:
			for j in T.columns:
				T.loc[i, j] = round(float(len(self.observations.loc[i][self.observations.loc[i]==j]))/len(self.observations.columns), 4)

		if self.true_estimates is not None:
			for i in self.true_estimates.index:
				T.loc[i] = self.true_estimates.loc[i]

		self.T = T
		return T


	def _cost(self, i, j):
		''' Naive cost function
			if i == j, cost = 0; else cost = 1
		'''
		if i == j:
			return 0
		else:
			return 1


	def _workers_cost(self):
		''' Estimate the expected cost of each worker self.C (Ipeirotis et al. 2010)
			Perfect workers will have zero cost while random workers/spammers will have high cost
			Ipeirotis et al. consider workers with cost < 0.5 to be a good quality worker
		'''
		wcost = []

		# Estimate workers' probability of assigning labels using eq. 2
		X = pd.DataFrame(index=self.workers, columns=self.labels)
		X = X.fillna(0)

		for k in X.index:
			for l in X.columns:
				for j in self.labels:
					X.loc[k, l] += self.em_pi.loc[k, j, l] * self.em_p[j]

		# For each worker k and label j, estimate the soft label vector (eq. 1) and workers' cost (eq. 3)
		for k in self.workers:
			c = 0
			for l in self.labels:
				soft = []
				for j in self.labels:
					soft.append((self.em_pi.loc[k, j, l] * self.em_p[j])/X.loc[k, l])

				c_soft = 0
				for i in self.labels: 
					for j in self.labels:
						if not math.isnan(soft[i]) and not math.isnan(soft[j]):
							c_soft += soft[i] * soft[j] * self._cost(i, j)

				c += c_soft * X.loc[k, l]
			wcost.append('%.4f' % c) 
		self.em_C = wcost


	def estimate(self, method, max_iteration=None, verbose=False):
		''' Estimate correct labels of objects using a specific algorithm
		Args:
			method: em for EM algorithm, mv for majority voting
			max_iteration: Maximum number of iterations for EM algorithm
		Return:
			Estimated correct labels
		'''
		if method=='em':
			if max_iteration is not None:
				estimators = self._em_estimator(None, max_iteration=max_iteration, verbose=verbose)
			else:
				estimators = self._em_estimator(None, verbose=verbose)

			# estimate the worker cost
			self._workers_cost()

			return estimators
		
		elif method=='mv':
			return self._mv_estimator()
		else:
			raise Exception('Error: Unrecognized method "'+method+'"')


if __name__ == '__main__':
	objects = np.arange(10)
	workers = np.arange(50)
	labels = np.arange(4)
	dist = [0.3, 0.4, 0.1, 0.1, 0.1] # [expert, normal, random, sloppy, uniform]
	gen = ObservationGenerator(objects, workers, labels, dist)
	observations = gen.obs
	est = LabelEstimator(observations, objects, workers, labels, None)
	t0 = time.time()
	T = est.estimate('em', verbose=True)
	t1 = time.time()
	print 'Done in %.4f sec' % (t1-t0)
	print 'Observation:\n%s' % observations
	print 'EM:\n%s' % T
	print 'True Labels:\n%s' % gen.true_labels
	print "Workers' types and costs"
	for i in workers:
		print '%s\t%s' % (gen.workers_types[i], est.em_C[i])
