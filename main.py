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
import odometry
import comparison
import fit_odometry

import numpy as np, matplotlib.pyplot as plt
import homere_control.io_dataset as iodata

if __name__ == '__main__':
    #Robot = Vehicle(0.08, 0.081, 0.2, 1, 4) #Rr, Rl, L, sigma_V, sigma_omega
    #ds = Robot.generate_data(-10, 10, 5000) #Vehicle + w_min, w_max, Nsample
    #ds = Robot.generate_outliers_uniform(-10, 10, 5000, 0.5, 0.5, 0.7, 2.2, 2.5, 2.5) #ratio_V, ratio_o, bias_V, bias_o, coef_V, coef_o


    #--------(Reality) Ackermann : Oscar-------------------------------------------------------------
    filename, type = './data/oscar_io_track_z.npz', 'oscar'
    #filename, type = './data/oscar_io_zigzag.npz', 'oscar'
    #filename, type = './data/oscar_io_figure_of_eight.npz', 'oscar'
    #filename, type = './data/oscar_io_oval.npz', 'oscar'
    #filename, type = './data/oscar_io_vel_sin_oval.npz', 'oscar'

    #--------(Reality) Diff Drive : Rosmip-----------------------------------------------------------
    #filename, type = './data/track_0.4mps.npz', 'oscar'
    #filename, type = './data/rosmip_z_foe_0.6ms.npz', 'oscar'
    #filename, type = './data/rosmip_z_foe_0.4ms.npz', 'oscar'
    #filename, type = './data/rosmip_z_foe_0.2ms.npz', 'oscar'
    #filename, type = './data/rosmip_z_foe_0.8ms.npz', 'oscar'
    #filename, type = './data/rosmip_z_oval01_0.4ms.npz', 'oscar'

    #--------(Simu Gazebo) Diff Drive : Rosmip-------------------------------------------------------
    #filename, type = './data/track_0.5mps_simu.npz', 'homere'
    #filename, type = './data/rosmip_z_foe_0.2ms_simu.npz', 'homere'


    ds, time = ut.data_converter(filename, type)
    initial_position = ut.data_initial_position(filename, type)

    """#pseudoinv
    a1, b1, c1, d1 = odo.coef_INV(ds)
    linear_formula1 = [a1, b1, c1, d1]
    print linear_formula1

    #RANSAC
    Rr_est2, Rl_est2 = odo.wheels_radius_RANSAC(ds)
    L_est2 = odo.space_wheels_RANSAC(ds, (Rr_est2, Rl_est2))
    print Rr_est2, Rl_est2, L_est2
    linear_formula2 = odo.converts_into_linear(Rr_est2, Rl_est2, L_est2)
    print linear_formula2

    #ANN (not use with weights)
    Rr_est3, Rl_est3, L_est3 = odo.all_param_AN(ds, odo.minkowski_loss)
    print Rr_est3, Rl_est3, L_est3
    linear_formula3 = odo.converts_into_linear(Rr_est3, Rl_est3, L_est3)
    print linear_formula3

    #filename, type = './data/track_0.5mps_simu.npz', 'homere'
    #filename, type = './data/rosmip_z_foe_0.2ms_simu.npz', 'homere'
    #ds, time = ut.data_converter(filename, type)
    #initial_position = ut.data_initial_position(filename, type)


    odometer1 = odometry.LinearOdometer(ds, time, linear_formula1)
    odometer1.compute_position(initial_position)

    odometer2 = odometry.LinearOdometer(ds, time, linear_formula2)
    odometer2.compute_position(initial_position)"""

    filename_fit, type_fit = './data/rosmip_z_foe_0.2ms_simu.npz', 'homere'
    ds_fit, time_fit = ut.data_converter(filename_fit, type_fit)
    odometer3 = odometry.NeuralNetworkOdometer(ds, time, fit_odometry.ann2, ds)
    odometer3.compute_position(initial_position)

    plt.figure()
    plt.title('track')
    ut.plot_true_position(filename, type)
    #odometer1.plot_position('odometry (pseudoinv)')
    #odometer2.plot_position('odometry (RANSAC)')
    odometer3.plot_position('odometry (simple ANN)')
    plt.legend()
    plt.show()