#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
import numpy as np

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def data_description(ds):
    """
        plot histogram for a dataset
    """
    wr = ds[0]
    wl = ds[1]
    V = ds[2]
    omega = ds[3]
    plt.subplot(2,2,1)
    plt.title('$\omega_r density$')
    plt.hist(wr, density=True, bins=10)
    plt.subplot(2,2,2)
    plt.title('$\omega_l density$')
    plt.hist(wl, density=True, bins=10)
    plt.subplot(2,2,3)
    plt.title('$V$ density')
    plt.hist(V, density=True, bins=10)
    plt.subplot(2,2,4)
    plt.title('$\Omega density$')
    plt.hist(omega, density=True, bins=10)

def plot3D(ds, step=4):
    """
        Plot a 3D scatter plot. 
        In : a ds=[wr, wl, V, omega] , step
        step : one point out step will be drawn (too slow animation otherwise)
    """
    wr = ds[0][::step]
    wl = ds[1][::step]
    V = ds[2][::step]
    omega = ds[3][::step]
    fig_V = plt.figure()
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax_V = Axes3D(fig_V)
    ax_V.scatter(wr, wl, V)
    fig_omega = plt.figure()
    ax_omega = Axes3D(fig_omega)
    ax_omega.scatter(wr, wl, omega)
    ax_V.set_xlabel("$\omega_r$")
    ax_V.set_ylabel("$\omega_l$")
    ax_V.set_zlabel("$V$")
    ax_omega.set_xlabel("$\omega_r$")
    ax_omega.set_ylabel("$\omega_l$")
    ax_omega.set_zlabel("$\Omega$")
    

def results_overlaid_on_data(ds, Rr_est, Rl_est, L_est, w_min, w_max, step=4):
    """
        plot a scatter plot of data with the estimated plane
        In : dataset, right radius, left radius, space, w_min, w_max, step
        All param are estimated param (it's the estimated plane, not the real one)
        w_min and w_max : to plot the plane as a 3D square
    """
    wr = ds[0][::step]
    wl = ds[1][::step]
    V = ds[2][::step]
    omega = ds[3][::step]
    V_regression = [(w_min, w_min,0.5*(Rr_est*w_min+Rl_est*w_min)),
        (w_max, w_min, 0.5*(Rr_est*w_max+Rl_est*w_min)),
        (w_max, w_max, 0.5*(Rr_est*w_max+Rl_est*w_max)),
        (w_min, w_max, 0.5*(Rr_est*w_min+Rl_est*w_max))]
    omega_regression = [(w_min, w_min,0.5*(Rr_est*w_min-Rl_est*w_min)/L_est),
        (w_max, w_min, 0.5*(Rr_est*w_max-Rl_est*w_min)/L_est),
        (w_max, w_max, 0.5*(Rr_est*w_max-Rl_est*w_max)/L_est),
        (w_min, w_max, 0.5*(Rr_est*w_min-Rl_est*w_max)/L_est)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Regression sur V")
    ax.set_zlabel("V")
    ax.set_ylabel("omega_l")
    ax.set_xlabel("omega_r")
    plane_V = mp3d.art3d.Poly3DCollection([V_regression], alpha=0.5, linewidth=1)
    plane_V.set_alpha = 0.5
    plane_V.set_facecolor("yellow")
    ax.add_collection3d(plane_V)
    ax.scatter(wr, wl, V, s=10)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Regression sur Omega")
    ax.set_zlabel("Omega")
    ax.set_ylabel("omega_l")
    ax.set_xlabel("omega_r")
    plane_omega = mp3d.art3d.Poly3DCollection([omega_regression], alpha=0.5, linewidth=1)
    plane_omega.set_alpha = 0.5
    plane_omega.set_facecolor("yellow")
    ax.add_collection3d(plane_omega)
    ax.scatter(wr, wl, omega)
