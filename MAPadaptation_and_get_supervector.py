# -*- coding: utf-8 -*-
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import copy
import matplotlib.pyplot as pl
"""
Created on Tue Dec 25 11:21:12 2018

@author: a-kojima

reference:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.338&rep=rep1&type=pdf

"""

class train_GMM:
    def returnGMM(features, NUMBER_OF_GAUSSIAN):
        gmm = GMM(n_components=NUMBER_OF_GAUSSIAN, covariance_type='diag')
        gmm.fit(features)
        return gmm
    
class opt:
    def normalize_meanvector(weight, var, mean_vec):
        normalize_mean = np.zeros(np.shape(mean_vec), dtype=np.float32)
        [NUMBER_OF_GAUSSIAN, FEATURE_ORDER] = np.shape(mean_vec)
        for ii in range(0, NUMBER_OF_GAUSSIAN):
            normalize_mean[ii, :] = np.sqrt(weight[ii]) * \
                            (1 / np.sqrt(var[ii, :])) * mean_vec[ii, :]
        return normalize_mean

# ===========================
# parameters    
# ===========================
NUMBER_OF_SAMPLE = 500
FEATURE_ORDER = 30
NUMBER_OF_GAUSSIAN = 4
RAND_VAL = 5
RAND_MEAN = 2
SCALING_FACTOR = 0.01

# generating samples
sample1 = np.random.randn(NUMBER_OF_SAMPLE, FEATURE_ORDER)
sample2 = np.random.randn(NUMBER_OF_SAMPLE, FEATURE_ORDER) * RAND_VAL + RAND_MEAN

# training init GMM
GMM_train_by_sample1 = train_GMM.returnGMM(sample1, NUMBER_OF_GAUSSIAN) 

# get posterior
probability = GMM_train_by_sample1.predict_proba(sample2)

# (8)
n_i = np.sum(probability, axis=0)

# (9)
E = np.zeros((FEATURE_ORDER, NUMBER_OF_GAUSSIAN), dtype=np.float32)
for ii in range(0, NUMBER_OF_GAUSSIAN):
    probability_gauss = np.tile(probability[:, ii],(FEATURE_ORDER, 1)).T * sample2
    E[:, ii] = np.sum(probability_gauss, axis=0) / n_i[ii]
 
# (14)    
alpha = n_i / (n_i + SCALING_FACTOR)
    
old_mean = copy.deepcopy(GMM_train_by_sample1.means_)
new_mean = np.zeros((NUMBER_OF_GAUSSIAN, FEATURE_ORDER), dtype=np.float32)

# (13)
for ii in range(0, NUMBER_OF_GAUSSIAN):
    new_mean[ii,:] = (alpha[ii] * E[:,ii]) + ((1 - alpha[ii]) * old_mean[ii, :])

# normalize
weight = GMM_train_by_sample1.weights_
var = GMM_train_by_sample1.covariances_

# get GMM supervector
norm_mean = opt.normalize_meanvector(weight, var, new_mean)
super_vector = np.reshape(norm_mean, NUMBER_OF_GAUSSIAN * FEATURE_ORDER)

norm_mean_old = opt.normalize_meanvector(weight, var,old_mean)
super_vector_old = np.reshape(norm_mean_old, NUMBER_OF_GAUSSIAN * FEATURE_ORDER)

pl.figure()
pl.plot(super_vector / np.sum(np.abs(super_vector)))
pl.plot(super_vector_old / np.sum(np.abs(super_vector_old)), 'r')
pl.legend(['adapted GMM supervector', 'original supervector'])

# ===========================
# confirm
# ===========================
sample2_many = np.random.randn(NUMBER_OF_SAMPLE * 100, FEATURE_ORDER) * RAND_VAL + RAND_MEAN
testGMM_many = train_GMM.returnGMM(sample2_many, NUMBER_OF_GAUSSIAN) 
testGMM_many_mean = testGMM_many.means_
pl.figure()
pl.plot(old_mean[:, 10], old_mean[:, 20], 'bo')
pl.plot(new_mean[:, 10], new_mean[:, 20], 'ro')
pl.plot(testGMM_many_mean[:, 10], testGMM_many_mean[:, 20], 'ko')
pl.legend(['before adapted ', 'after adapted', 'true'])
pl.grid(True)
pl.show()