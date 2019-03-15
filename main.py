#!/usr/bin/env python
# -*- coding: utf-8 -*-



from __future__ import division

import sys
import keras, pickle
import keras.backend as K 
import pdb
import sklearn.linear_model

from vehicle import Vehicle
import description as descr
import myutils as ut
import odometry_test as odo
import comparison

import numpy as np, matplotlib.pyplot as plt
import homere_control.io_dataset as iodata

if __name__ == '__main__':
    #Robot = Vehicle(0.08, 0.081, 0.2, 1, 4) #Rr, Rl, L, sigma_V, sigma_omega
    #ds = Robot.generate_data(-10, 10, 5000) #Vehicle + w_min, w_max, Nsample
    #ds = Robot.generate_outliers_uniform(-10, 10, 5000, 0.5, 0.5, 0.7, 2.2, 2.5, 2.5) #ratio_V, ratio_o, bias_V, bias_o, coef_V, coef_o


    filename, type = './data/oscar_io_oval.npz', 'oscar'
    #filename, type = '/home/poine/work/homere/homere_control/data/rosmip/gazebo/rosmip_io_02.npz', 'rosmip'
    #filename, type = '/home/poine/work/oscar/oscar/oscar_control/paths/enac_bench/path_01.npz', 'rosmip'
    ds = ut.data_converter(filename, type)[0]


    #descr.plot3D(ds, step=10) #step=4 (by default)
    #plt.show()
    
    a, b, c, d = odo.coef_INV(ds)
    Rr_est, Rl_est = odo.wheels_radius_INV(ds)
    L_est = odo.space_wheels_INV(ds, (Rr_est, Rl_est))

    res = odo.residuals(ds, Rr_est, Rl_est, L_est)
    res2 = odo.residuals2(ds, a, b, c, d)

    print "---------|-----mu-----|-----sigma-------|"
    print "-----------------------------------------"
    print "meth1----|  ", res[1][1],"   |   ", res[1][2]
    print "-----------------------------------------"
    print "meth2----|  ", res2[1][1],"   |   ", res2[1][2]
    #res = odo.residuals(ds, Rr_est, Rl_est, L_est)
    #print res

    #comparison.comparison_noise(1, 4) #sigma_V_max, sigma_omega_max
    #plt.show()

    """file, type = '/home/poine/work/homere/homere_control/data/homere_io_10.npz', 'homere'
    if len(sys.argv) > 1: file = sys.argv[1]
    ds = ut.data_converter(file, type)
    print odo.all_param_AN(ds)"""

    #Rr_est, Rl_est, L_est = odo.all_param_AN(ds)
    #comparison.comparaison_minkowski(ds, 10)

    plt.show()

    #descr.plot3D(ds, step=10)
    #plt.show()
    #odo.all_methods(ds)


    #plt.figure()
    #descr.data_description(ds)
    #descr.plot3D(ds, 4)
    #descr.results_overlaid_on_data(ds, Rr_est, Rl_est, L_est, -10, 10)
    #plt.show()"""

    

