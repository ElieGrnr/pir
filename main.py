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

import numpy as np, matplotlib.pyplot as plt
import homere_control.io_dataset as iodata

if __name__ == '__main__':
    #Robot = Vehicle(0.08, 0.081, 0.2, 1, 4) #Rr, Rl, L, sigma_V, sigma_omega
    #ds = Robot.generate_data(-10, 10, 5000) #Vehicle + w_min, w_max, Nsample
    #ds = Robot.generate_outliers_uniform(-10, 10, 5000, 0.5, 0.5, 0.7, 2.2, 2.5, 2.5) #ratio_V, ratio_o, bias_V, bias_o, coef_V, coef_o


    #filename, type = './data/oscar_io_track_z.npz', 'oscar'
    #filename, type = './data/oscar_io_zigzag.npz', 'oscar'
    filename, type = './data/oscar_io_figure_of_eight.npz', 'oscar'
    #filename, type = './data/oscar_io_oval.npz', 'oscar'
    #filename, type = './data/oscar_io_vel_sin_oval.npz', 'oscar'

    ds, time = ut.data_converter(filename, type)
    print(time[-1]-time[0])
    initial_position = ut.data_initial_position(filename, type)

    #pseudoinv
    a1, b1, c1, d1 = odo.coef_INV(ds)
    linear_formula1 = [a1, b1, c1, d1]
    print linear_formula1

    #RANSAC
    Rr_est, Rl_est = odo.wheels_radius_RANSAC(ds)
    L_est = odo.space_wheels_RANSAC(ds, (Rr_est, Rl_est))
    linear_formula2 = odo.converts_into_linear(Rr_est, Rl_est, L_est)
    print linear_formula2

    #ANN
    Rr_est, Rl_est, L_est = odo.all_param_AN(ds)
    linear_formula3 = odo.converts_into_linear(Rr_est, Rl_est, L_est)
    print linear_formula3

    #Rr_est, Rl_est = odo.wheels_radius_INV(ds)
    #L_est = odo.space_wheels_INV(ds, (Rr_est, Rl_est))

    #res = odo.residuals(ds, Rr_est, Rl_est, L_est)
    #res2 = odo.residuals2(ds, a, b, c, d)

    odometer1 = odometry.LinearOdometer(ds, time, linear_formula1)
    odometer1.compute_position(initial_position)

    odometer2 = odometry.LinearOdometer(ds, time, linear_formula2)
    odometer2.compute_position(initial_position)

    odometer3 = odometry.LinearOdometer(ds, time, linear_formula3)
    odometer3.compute_position(initial_position)

    plt.figure()
    plt.title('figure_of_eight')
    ut.plot_truth(filename, type)
    odometer1.plot('odometry (pseudoinv)')
    odometer2.plot('odometry (RANSAC)')
    odometer3.plot('odometry (simple ANN)')
    plt.legend()
    plt.show()

    

