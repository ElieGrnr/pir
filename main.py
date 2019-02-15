#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division


import keras, pickle
import keras.backend as K 
import pdb
import sklearn.linear_model

from vehicle import Vehicle
import description as descr
import myutils as ut
import odometry_test as odo
import numpy as np, matplotlib.pyplot as plt
import homere_control.io_dataset as iodata

if __name__ == '__main__':
    #Robot = Vehicle(0.08, 0.081, 0.2, 0.1, 0.5) #Rr, Rl, L, sigma_V, sigma_omega
    #ds = Robot.generate_random_data(-10, 10, 5000, True, 0.5, 0.5, 2, 2) #wmin, wmax, nbsample, outliers, 
    #ration_V, ratio_omega, coef_V, coef_omega

    filename, type = '/home/poine/work/homere/homere_control/data/homere_io_10.npz', 'homere'
    ds = ut.data_converter(filename, type)


    descr.plot3D(ds, step=20) #step=4 (by default)
    plt.show()

    print "\nPseudo-inverse : "
    Rr_est, Rl_est = odo.wheels_radius_INV(ds)
    L_est = odo.space_wheels_INV(ds, (Rr_est, Rl_est))
    print "-------Rayon roue droite estimé : ", Rr_est
    print "-------Rayon roue gauche estimé : ", Rl_est
    print "-------Voie estimée : ", L_est

    print("\nRANSAC : ")
    Rr_est, Rl_est = odo.wheels_radius_RANSAC(ds)
    L_est = odo.space_wheels_RANSAC(ds, (Rr_est, Rl_est))
    print "-------Rayon roue droite estimé : ", Rr_est
    print "-------Rayon roue gauche estimé : ", Rl_est
    print "-------Voie estimée : ", L_est

    print("\nRéseau de neurones : ")
    Rr_est, Rl_est, L_est = odo.all_param_AN(ds, odo.minkowski_loss)
    print "-------Rayon roue droite estimé : ", Rr_est
    print "-------Rayon roue gauche estimé : ", Rl_est
    print "-------Voie estimée : ", L_est

    #plt.figure()
    #descr.data_description(ds)
    #descr.plot3D(ds, 4)
    #descr.results_overlaid_on_data(ds, Rr_est, Rl_est, L_est, -10, 10)
    #plt.show()"""

    

