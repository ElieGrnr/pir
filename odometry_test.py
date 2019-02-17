#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import os, logging, numpy as np, matplotlib.pyplot as plt
import keras, pickle
import keras.backend as K 
import pdb
import sklearn.linear_model
from vehicle import Vehicle
import description as descr
import myutils as ut

import homere_control.io_dataset as iodata

def minkowski_loss(y_true, y_pred):
    """
        Compute Minkowski's loss for AN. Better than MSE if outliers (maybe not).
    """
    r = 1.5 #often 0.4
    return K.mean(K.pow(K.abs(y_pred-y_true),r))

def wheels_radius_AN(ds, myloss='mean_squared_error'):
    """
        Compute the wheels radius of both wheels, presumed different, according a AN (no hidden layer)
        In : ds, a data set ds = [wr, wl, V, omega]
        Out : right radius estimated, left radius estimated
    """
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    input = np.zeros((len(wl),2))
    input[:,0] = wr
    input[:,1] = wl

    input_layer = keras.layers.Input((2,),name="input") #wr et wl
    hidden_layer = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform',
                             input_shape=(2,), use_bias=False, name="output")
    output_layer = hidden_layer(input_layer)
    ann = keras.models.Model(inputs=input_layer, outputs=output_layer)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    ann.compile(loss=myloss(1.5), optimizer=opt)
    ann_in, ann_out = input, V
    history = ann.fit(ann_in, ann_out, epochs=30, batch_size=64,  verbose=0,
                   shuffle=True, validation_split=0.1)#, callbacks=callbacks)
    weights = hidden_layer.get_weights()
    Rr_est = 2*weights[0][0]
    Rl_est = 2*weights[0][1]
    return Rr_est[0], Rl_est[0]

def space_wheels_AN(ds, R_est, myloss='mean_squared_error'):
    """
        Compute the space wheel according a AN (no hidden layer)
        In : ds (a data set ds = [wr, wl, V, omega]) , estimated radius (a couple (rights, left))
        Out : space wheel estimated
    """
    wr = ds[0]
    wl = ds[1]
    omega = ds[3]
    Rl_est, Rr_est = R_est[1], R_est[0]
    input = np.zeros((len(wl),2))
    input[:,0] = wr
    input[:,1] = wl
    input_layer = keras.layers.Input((2,),name="input") #wr et wl
    hidden_layer = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform',
                             input_shape=(2,), use_bias=False, name="output")
    output_layer = hidden_layer(input_layer)
    ann = keras.models.Model(inputs=input_layer, outputs=output_layer)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    ann.compile(loss=myloss, optimizer=opt)
    ann_in, ann_out = input, omega
    history = ann.fit(ann_in, ann_out, epochs=30, batch_size=64,  verbose=0,
                   shuffle=True, validation_split=0.1)#, callbacks=callbacks)
    weights = hidden_layer.get_weights()
    L_est1 = 1/(2*weights[0][0]/Rr_est)
    L_est2 = -1/(2*weights[0][1]/Rl_est)
    L_est = (L_est1+L_est2)/2   #moyenne des deux longueurs obtenues
    return L_est[0]

def all_param_AN(ds, myloss='mean_squared_error'):
    """
        Compute the wheels radius of both wheels and the space wheel according a single AN (no hidden layer)
        In : ds (a data set ds = [wr, wl, V, omega])
        Out : right radius, left radius, space wheel
    """
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    omega = ds[3]
    input = np.zeros((len(wl),2))
    input[:,0] = wr
    input[:,1] = wl
    output = np.zeros((len(wl),2))
    output[:,0] = V
    output[:,1] = omega
    input_layer = keras.layers.Input((2,),name="input") #wr et wl
    hidden_layer = keras.layers.Dense(2, activation='linear', kernel_initializer='uniform',
                             input_shape=(2,), use_bias=False, name="output") #V et omega
    output_layer = hidden_layer(input_layer)
    ann = keras.models.Model(inputs=input_layer, outputs=output_layer)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    ann.compile(loss=myloss, optimizer=opt)
    ann_in, ann_out = input, output
    history = ann.fit(ann_in, ann_out, epochs=40, batch_size=64,  verbose=0,
                   shuffle=True, validation_split=0.1)#, callbacks=callbacks)

    """plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()"""

    weights = hidden_layer.get_weights()[0]
    Rr_est = weights[0][0]*2
    Rl_est = weights[1][0]*2
    L_est1 = 1/(2*weights[0][1]/Rr_est)
    L_est2 = -1/(2*weights[1][1]/Rr_est)
    return Rr_est, Rl_est, (L_est2+L_est1)/2 #moyenne des deux longueurs obtenues

def wheels_radius_INV(ds):
    """
        Compute the wheels radius of both wheels according a regression X=pseudoinv(H)*V
        In : ds (a data set ds = [wr, wl, V, omega])
        Out : right radius, left radius, space wheel
    """
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    Nsample = len(wr)
    H = np.zeros((Nsample,2))
    H[:,0] = wr*0.5
    H[:,1] = wl*0.5 
    X = np.dot(np.linalg.pinv(H),V) #X=rayons estimés
    Rl_est, Rr_est = X[1], X[0]
    return Rr_est, Rl_est

def space_wheels_INV(ds, R_est):
    """
        Compute the space wheel according a regression X=pseudoinv(H)*omega
        In : ds (a data set ds = [wr, wl, V, omega]) , estimated radius (a couple (rights, left))
        Out : wheel space estimated
    """
    wr = ds[0]
    wl = ds[1]
    omega = ds[3]
    Rl_est, Rr_est = R_est[1], R_est[0]
    Nsample = len(ds[0])
    H = np.zeros((Nsample,1))
    H[:,0] = 0.5*(Rr_est*wr-Rl_est*(wl)) 
    X = np.dot(np.linalg.pinv(H),omega)
    L_est = 1/X[0]
    return L_est

def wheels_radius_RANSAC(ds):
    """
        Compute the wheels radius of both wheels according a regression RANSAC
        In : ds (a data set ds = [wr, wl, V, omega])
        Out : right radius, left radius, space wheel
    """
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    Nsample = len(wr)
    H = np.zeros((Nsample,2))
    H[:,0] = wr*0.5
    H[:,1] = wl*0.5
    ransac_radius = sklearn.linear_model.RANSACRegressor(base_estimator=sklearn.linear_model.LinearRegression(fit_intercept=False))
    ransac_radius.fit(H, V)
    [Rr_est, Rl_est] = ransac_radius.estimator_.coef_
    return Rr_est, Rl_est

def space_wheels_RANSAC(ds, R_est):
    """
        Compute the space wheel according a regression RANSAC
        In : ds (a data set ds = [wr, wl, V, omega]) , estimated radius (a couple (right, left))
        Out : wheel space estimated
    """
    wr = ds[0]
    wl = ds[1]
    omega = ds[3]
    Rr_est, Rl_est = R_est[0], R_est[1]
    Nsample = len(ds[0])
    H = np.zeros((Nsample,1))
    H[:,0] = 0.5*(Rr_est*wr-Rl_est*(wl)) 
    ransac_space = sklearn.linear_model.RANSACRegressor(base_estimator=sklearn.linear_model.LinearRegression(fit_intercept=False))
    ransac_space.fit(H, omega)
    L_est = 1/ransac_space.estimator_.coef_[0]
    return L_est

def residuals(ds, Rr_est, Rl_est, L_est):
    """
        Computes residuals: Vreal-Vestimated and omega_real - omega_estimated
        Gives sigma and mu. (std and mean)
    """
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    omega = ds[3]
    res_V = V - 0.5*(Rr_est*wr+Rl_est*wl) #res=Vréel - Vestimé
    res_omega = omega - (0.5/L_est)*(Rr_est*wr-Rl_est*wl)  #res=Oréel-Oestimé

    res_V_sigma, res_V_mu = np.std(res_V), np.mean(res_V)
    res_omega_sigma, res_omega_mu = np.std(res_omega), np.mean(res_omega)

    return [[res_V, res_V_mu, res_V_sigma], [res_omega, res_omega_mu, res_omega_sigma]]

def all_methods(ds):
    print "\nPseudo-inverse : "
    Rr_est, Rl_est = wheels_radius_INV(ds)
    L_est = space_wheels_INV(ds, (Rr_est, Rl_est))
    print "-------Rayon roue droite estimé : ", Rr_est
    print "-------Rayon roue gauche estimé : ", Rl_est
    print "-------Voie estimée : ", L_est

    print("\nRANSAC : ")
    Rr_est, Rl_est = wheels_radius_RANSAC(ds)
    L_est = space_wheels_RANSAC(ds, (Rr_est, Rl_est))
    print "-------Rayon roue droite estimé : ", Rr_est
    print "-------Rayon roue gauche estimé : ", Rl_est
    print "-------Voie estimée : ", L_est

    print("\nRéseau de neurones : ")
    Rr_est, Rl_est, L_est = all_param_AN(ds, minkowski_loss)
    print "-------Rayon roue droite estimé : ", Rr_est
    print "-------Rayon roue gauche estimé : ", Rl_est
    print "-------Voie estimée : ", L_est
