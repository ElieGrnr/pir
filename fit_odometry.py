#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

#import os, logging, numpy as np, matplotlib.pyplot as plt
#import keras, pickle
#import keras.backend as K 
#import pdb
#import sklearn.linear_model
import myutils as ut

import keras
	
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

filename, type = './data/track_0.5mps_simu.npz', 'homere'
ds, time = ut.data_converter(filename, type)


def make_io_ann1(ds):
	wr = ds[0]
	wl = ds[1]
	V = ds[2]
	omega = ds[3]
	ann_in = np.zeros((len(wl),2))
	ann_in[:,0] = wr
	ann_in[:,1] = wl
	ann_out = np.zeros((len(wl),2))
	ann_out[:,0] = V
	ann_out[:,1] = omega
	return ann_in, ann_out

def ann1(ds, ds_fit):
    """
        Compute the wheels radius of both wheels and the space wheel according a single AN (no hidden layer)
        In : ds (a data set ds = [wr, wl, V, omega])
        Out : right radius, left radius, space wheel
    """

    myloss='mean_squared_error'

    ann_in, ann_out = make_io_ann1(ds_fit)
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='linear', kernel_initializer='uniform', use_bias=False))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=myloss, optimizer=opt)
    history = model.fit(ann_in, ann_out, epochs=40, batch_size=64,  verbose=0, shuffle=True, validation_split=0.1)

    X, _ = make_io_ann1(ds)
    velocity = model.predict(X)

    """plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()"""

    return velocity



def make_io_ann2(ds):
	wr = ds[0]
	wl = ds[1]
	V = ds[2]
	omega = ds[3]

	ann_in = np.zeros((len(wl),4))
	ann_in[:,0] = wr
	ann_in[:,1] = wl
	ann_in[:,2] = np.power(wr,2)
	ann_in[:,3] = np.power(wl,2)
	"""ann_in[0,4] = 0
	ann_in[0,5] = 0
	ann_in[1:,4] = V[:len(V)-1]
	ann_in[1:,5] = omega[:len(omega)-1]"""

	ann_out = np.zeros((len(V),2))
	ann_out[:,0] = V
	ann_out[:,1] = omega
	return ann_in, ann_out

def ann2(ds, ds_fit):

	ann_in, ann_out = make_io_ann2(ds_fit)

	model = Sequential()
	model.add(Dense(10, input_dim=4, activation='linear', kernel_initializer='uniform', use_bias=True))
	#model.add(Dense(10, activation='relu', kernel_initializer='uniform', use_bias=True))
	#model.add(Dense(10, activation='relu', kernel_initializer='uniform', use_bias=True))
	model.add(Dense(10, activation='linear', kernel_initializer='uniform', use_bias=True))
	model.add(Dense(2, activation='linear', kernel_initializer='uniform', use_bias=True))
	opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='mean_squared_error', optimizer=opt)
	model.fit(ann_in, ann_out, epochs=100, batch_size=64,  verbose=1, shuffle=True, validation_split=0.1)

	X, _ = make_io_ann2(ds)
	velocity = model.predict(X)



	"""layer = Dense(2, input_dim=6, activation='relu', kernel_initializer='uniform', use_bias=True)
	model.add(layer)
	opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='mean_squared_error', optimizer=opt)
	model.fit(X, Y, epochs=100, batch_size=64,  verbose=1, shuffle=True, validation_split=0.1)
	scores = model.evaluate(X, Y)
	weights = layer.get_weights()[0]
	Rr_est = weights[0][0]*2
	Rl_est = weights[1][0]*2
	L_est1 = 1/(weights[0][1]/Rr_est)
	L_est2 = -1/(weights[1][1]/Rr_est)
	print Rr_est, Rl_est, L_est1"""

	return velocity
 


