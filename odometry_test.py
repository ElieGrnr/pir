#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt
import keras, pickle
import pdb
from vehicle import Vehicle
import description as descr



def wheels_radius_AN(ds):
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
    ann.compile(loss='mean_squared_error', optimizer=opt)
    ann_in, ann_out = input, V
    history = ann.fit(ann_in, ann_out, epochs=30, batch_size=64,  verbose=1,
                   shuffle=True, validation_split=0.1)#, callbacks=callbacks)
    weights = hidden_layer.get_weights()
    Rr_est = 2*weights[0][0]
    Rl_est = 2*weights[0][1]
    return Rr_est[0], Rl_est[0]

def space_wheels_AN(ds, R_est):
    wr = ds[0]
    wl = ds[1]
    omega = ds[3]
    Rl_est, Rr_est = R_est[0], R_est[1]
    input = np.zeros((len(wl),2))
    input[:,0] = wr
    input[:,1] = wl
    input_layer = keras.layers.Input((2,),name="input") #wr et wl
    hidden_layer = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform',
                             input_shape=(2,), use_bias=False, name="output")
    output_layer = hidden_layer(input_layer)
    ann = keras.models.Model(inputs=input_layer, outputs=output_layer)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    ann.compile(loss='mean_squared_error', optimizer=opt)
    ann_in, ann_out = input, omega
    history = ann.fit(ann_in, ann_out, epochs=30, batch_size=64,  verbose=1,
                   shuffle=True, validation_split=0.1)#, callbacks=callbacks)
    weights = hidden_layer.get_weights()
    L_est1 = 1/(2*weights[0][0]/Rr_est)
    L_est2 = -1/(2*weights[0][1]/Rl_est)
    L_est = (L_est1+L_est2)/2
    return L_est[0]

def all_param_AN(ds):
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
                             input_shape=(2,), use_bias=False, name="output")
    output_layer = hidden_layer(input_layer)
    ann = keras.models.Model(inputs=input_layer, outputs=output_layer)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    ann.compile(loss='mean_squared_error', optimizer=opt)
    ann_in, ann_out = input, output
    history = ann.fit(ann_in, ann_out, epochs=30, batch_size=64,  verbose=1,
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
    return Rr_est, Rl_est, (L_est2+L_est1)/2

def wheels_radius_INV(ds):
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    Nsample = len(wr)
    H1 = np.zeros((Nsample,2))
    H1[:,0] = wr*0.5
    H1[:,1] = wl*0.5
    X1 = np.dot(np.linalg.pinv(H1),V)
    Rl_est, Rr_est = X1[0], X1[1]
    return Rl_est, Rr_est

def space_wheels_INV(ds, R_est):
    wr = ds[0]
    wl = ds[1]
    omega = ds[3]
    Rl_est, Rr_est = R_est[0], R_est[1]
    Nsample = len(ds[0])
    H2 = np.zeros((Nsample,1))
    H2[:,0] = 0.5*(Rr_est*wr-Rl_est*(wl)) 
    X2 = np.dot(np.linalg.pinv(H2),omega)
    L_est = 1/X2[0]
    return L_est




if __name__ == '__main__':
    Robot = Vehicle(0.08, 0.081, 0.2, 0.01, 0.01)
    ds = Robot.generate_random_data(-10, 10, 5000, True)
    #Rr_est, Rl_est = wheels_radius_AN(ds)
    #L_est = space_wheels_AN(ds, (Rr_est, Rl_est))
    #Rr_est, Rl_est = wheels_radius_INV(ds)
    #L_est = space_wheels_INV(ds, R_est)
    Rr_est, Rl_est, L_est = all_param_AN(ds)
    print("Rayon roue droite estimé : ", Rr_est)
    print("Rayon roue gauche estimé : ", Rl_est)
    print("Voie estimée : ", L_est) 

    #plt.figure()
    #descr.data_description(ds)
    #descr.plot3D(ds, 4)
    descr.results_overlaid_on_data(ds, Rr_est, Rl_est, L_est, -10, 10)
    plt.show()


