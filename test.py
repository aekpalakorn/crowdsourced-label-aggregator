import argparse
import numpy as np
import time
import estimator


parser = argparse.ArgumentParser(description='Example of true label estimation from synthetic data')
parser.add_argument('-o', '--objects', type=int, default=10, help='Number of objects')
parser.add_argument('-w', '--workers', type=int, default=50, help='Number of workers')
parser.add_argument('-l', '--labels', type=int, default=4, help='Number of labels')
parser.add_argument('-d', '--distribution', type=float, nargs='+', default=[0.2, 0.2, 0.2, 0.2, 0.2], help='Worker-type distribution (expert, normal, random, sloppy, uniform)')
args = parser.parse_args()
cmd_args = vars(args)

objects = np.arange(cmd_args['objects'])
workers = np.arange(cmd_args['workers'])
labels = np.arange(cmd_args['labels'])
dist = cmd_args['distribution']

print 'Parameters: Object=%s, workers=%s, labels=%s, worker-type (expert, normal, random, sloppy, uniform) distribution=%s' % (len(objects), len(workers), len(labels), dist)

gen = estimator.ObservationGenerator(objects, workers, labels, dist)
observations = gen.obs
est = estimator.LabelEstimator(observations, objects, workers, labels, None)
t0 = time.time()
T = est.estimate('em', verbose=True)
t1 = time.time()

print 'Done in %.4f sec' % (t1-t0)
print 'Observation:\n%s\n' % observations
print 'EM:\n%s\n' % T
print 'True Labels:\n%s\n' % gen.true_labels
print "Workers' types and costs"
for i in workers:
	print '%s\t%s' % (gen.workers_types[i], est.em_C[i])