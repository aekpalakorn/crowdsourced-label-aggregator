python-label-aggregator
=======================

Estimate true labels of objects and qualities (costs) of workers from multiple workers in crowdsourcing tasks using EM algorithm (Dawid and Skene, 1979; Ipeirotis et al. 2010). Test data are generated based on the framework by Nguyen et al. 2013.


Requirements
------------
* Python 2.7


Dependencies
------------
* Numpy 1.8.0
* Pandas 0.12.0


Usage
-----
Run `python estimator.py` 

The example generates synthethic observations given the lists of objects, workers, labels, and worker-type distribution as parameters. Then, it estimates the true labels from the test data using the EM algorithm.


References
----------
1. A. P. Dawid and  A. M. Skene (1979) Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. Journal of the Royal Statistical Society. Series C (Applied Statistis), 28:1, pp. 20-28
2. P. G. Ipeirotis et al. (2010) Quality Management on Amazon Mechanical Turk. In. Proc. of HCOMP.
3. Q. V. H. Nguyen et al. (2013) An Evaluation of Aggregation Techniques in Crowdsourcing. In Proc. of WISE.
