#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstrates how the partial likelihood from a Cox proportional hazards
model can be used in a NN loss function. An example shows how a NN with
one linear-activation layer and the (negative) log partial likelihood as
loss function produces approximately the same predictor weights as a Cox
model fit in a more conventional way.
"""
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_kidney_transplant
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras.backend as K

# Use example dataset from lifelines module
kidtx = load_kidney_transplant()
# First three rows:
#   time  death  age  black_male  white_male  black_female
#0     1      0   46           0           1             0
#1     5      0   51           0           1             0
#2     7      1   55           0           1             0
X = kidtx.drop(["time", "death"], axis = 1).values
y = np.transpose(np.array((kidtx["time"], kidtx["death"])))
n = y.shape[0]

# Build model structure
model = Sequential()
model.add(Dense(units = 1, activation = "linear", use_bias = False, input_shape=[4]))

# Define loss function
# y_true = (n x 2) array with y_true[i, 0] the survival time
#          for individual i and y_true[i, 1] the event indicator
# y_pred = (n x 1) array of linear predictor (x * beta) values
def neg_log_pl(y_true, y_pred):
    # Sort by survival time (descending) so that
    # - If there are no tied survival times, the risk set
    #   for event i is individuals 0 through i
    # - If there are ties, and time[i - k] through time[i]
    #   represent all times equal to time[i], then the risk set
    #   for events i - k through i is individuals 0 through i
    sorting = tf.nn.top_k(y_true[:, 0], k = n)
    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices)
    risk = K.exp(xbeta)
    # For each set of tied survival times, put the sum of the
    # corresponding risk (exp[x * beta]) values at the first
    # position in the sorted array of times while setting other
    # positions to 0 so that the cumsum operation will result
    # in each of the positions having the same sum of risks
    for i in range(time.shape[0] - 1, 0, -1):
        # Going from smallest survival times to largest
        if time[i] == time[i - 1]:
            # Push risk to the later time (earlier in array position)
            risk[i - 1] = risk[i - 1] + risk[i]
            risk[i] = 0
    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk)
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms
    return -K.sum(loglik)

# Compile model
model.compile(optimizer = "adam", loss = neg_log_pl)

# Fit model with the whole dataset as a batch, since the
# partial likelihood depends on all observations
model.fit(X, y, batch_size = n, epochs = 3000)

# Compare to Cox model
cph = CoxPHFitter()
# CoxPHFitter uses Efron's method for handling tied survival times,
# whereas neg_log_pl uses Breslow's method, so the likelihood
# functions being optimized are not exactly the same
cph.fit(kidtx, duration_col = "time", event_col = "death")
cph.print_summary(decimals=8)
model.get_weights()