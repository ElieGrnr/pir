#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os, logging, numpy as np, matplotlib.pyplot as plt
import keras, pickle
import keras.backend as K
import pdb
import odometry_test as odo
from vehicle import Vehicle

#-------------------------------------Relevant analysis below--------------------------------------


def comparison_ANs(sigma_V_max, sigma_omega_max):
	"""
		draws a comparison of two-AN algo and all_param_An algo according noise. No outliers.
		sigma(noise)_V = [0, ..., sigma_V_max]
		sigma(noise)_omega = [0, ..., sigma_omega_max]
		Comparison based on mean and std of residuals. (fct odometry.residuals())
		Results : no difference !
	"""
	Npoints = 5 #nb of point for the plot
	tab_sigma_V = np.linspace(0, sigma_V_max, Npoints)
	tab_sigma_omega = np.linspace(0, sigma_omega_max, Npoints)

	res_V_sigma1 = np.zeros(Npoints)
	res_V_mu1 = np.zeros(Npoints)
	res_omega_sigma1 = np.zeros(Npoints)
	res_omega_mu1 = np.zeros(Npoints)
	res_V_sigma2 = np.zeros(Npoints)
	res_V_mu2 = np.zeros(Npoints)
	res_omega_sigma2 = np.zeros(Npoints)
	res_omega_mu2 = np.zeros(Npoints)

	for i in range(Npoints):

		Robot = Vehicle(0.08, 0.081, 0.2, tab_sigma_V[i], tab_sigma_omega[i])

		ds = Robot.generate_data(-10, 10, 5000)

		Rr_est1, Rl_est1 = odo.wheels_radius_AN(ds)
		L_est1 = odo.space_wheels_AN(ds, (Rr_est1, Rl_est1))
		residuals1 = odo.residuals(ds, Rr_est1, Rl_est1, L_est1)
		res_V_sigma1[i] = residuals1[0][2]
		res_V_mu1[i] = residuals1[0][1]
		res_omega_sigma1[i] = residuals1[1][2]
		res_omega_mu1[i] = residuals1[1][1]

		Rr_est2, Rl_est2, L_est2 = odo.all_param_AN(ds)
		residuals2 = odo.residuals(ds, Rr_est2, Rl_est2, L_est2)
		res_V_sigma2[i] = residuals2[0][2]
		res_V_mu2[i] = residuals2[0][1]
		res_omega_sigma2[i] = residuals2[1][2]
		res_omega_mu2[i] = residuals2[1][1]

	plt.figure()
	plt.subplot(221)
	plt.plot(tab_sigma_V, res_V_sigma1, label="2ANs")
	plt.plot(tab_sigma_V, res_V_sigma2, label="1ANs")
	plt.xlabel("$\sigma_V$")
	plt.ylabel("ecart type")
	plt.title("V residuals")
	plt.legend()
	plt.subplot(222)
	plt.plot(tab_sigma_V, res_V_mu1)
	plt.plot(tab_sigma_V, res_V_mu2)
	plt.xlabel("$\sigma_V$")
	plt.ylabel("moyenne")
	plt.title("V residuals")
	plt.subplot(223)
	plt.plot(tab_sigma_omega, res_omega_sigma1)
	plt.plot(tab_sigma_omega, res_omega_sigma2)
	plt.xlabel("$\sigma_{\omega}$")
	plt.ylabel("ecart type")
	plt.title("omega residuals")
	plt.subplot(224)
	plt.plot(tab_sigma_omega, res_omega_mu1)
	plt.plot(tab_sigma_omega, res_omega_mu2)
	plt.xlabel("$\sigma_{\omega}$")
	plt.ylabel("moyene")
	plt.title("omega residuals")


def comparison_noise(sigma_V_max, sigma_omega_max):
	Npoints = 5 #nb of point for the plot
	tab_sigma_V = np.linspace(0, sigma_V_max, Npoints)
	tab_sigma_omega = np.linspace(0, sigma_omega_max, Npoints)

	res_V_sigma1 = np.zeros(Npoints)
	res_V_mu1 = np.zeros(Npoints)
	res_omega_sigma1 = np.zeros(Npoints)
	res_omega_mu1 = np.zeros(Npoints)
	res_V_sigma2 = np.zeros(Npoints)
	res_V_mu2 = np.zeros(Npoints)
	res_omega_sigma2 = np.zeros(Npoints)
	res_omega_mu2 = np.zeros(Npoints)

	for i in range(Npoints):

		Robot = Vehicle(0.08, 0.081, 0.2, tab_sigma_V[i], tab_sigma_omega[i])

		ds = Robot.generate_data(-10, 10, 5000)

		Rr_est1, Rl_est1, L_est1 = odo.all_param_AN(ds)
		residuals1 = odo.residuals(ds, Rr_est1, Rl_est1, L_est1)
		res_V_sigma1[i] = residuals1[0][2]
		res_V_mu1[i] = residuals1[0][1]
		res_omega_sigma1[i] = residuals1[1][2]
		res_omega_mu1[i] = residuals1[1][1]

		Rr_est2, Rl_est2 = odo.wheels_radius_INV(ds)
		L_est2 = odo.space_wheels_INV(ds, (Rr_est2, Rl_est2))
		residuals2 = odo.residuals(ds, Rr_est2, Rl_est2, L_est2)
		res_V_sigma2[i] = residuals2[0][2]
		res_V_mu2[i] = residuals2[0][1]
		res_omega_sigma2[i] = residuals2[1][2]
		res_omega_mu2[i] = residuals2[1][1]

	plt.figure()
	plt.subplot(221)
	plt.plot(tab_sigma_V, res_V_sigma1, label="AN")
	plt.plot(tab_sigma_V, res_V_sigma2, label="INV")
	plt.xlabel("$\sigma_V$")
	plt.ylabel("ecart type")
	plt.title("V residuals")
	plt.legend()
	plt.subplot(222)
	plt.plot(tab_sigma_V, res_V_mu1)
	plt.plot(tab_sigma_V, res_V_mu2)
	plt.xlabel("$\sigma_V$")
	plt.ylabel("moyenne")
	plt.title("V residuals")
	plt.subplot(223)
	plt.plot(tab_sigma_omega, res_omega_sigma1)
	plt.plot(tab_sigma_omega, res_omega_sigma2)
	plt.xlabel("$\sigma_{\omega}$")
	plt.ylabel("ecart type")
	plt.title("omega residuals")
	plt.subplot(224)
	plt.plot(tab_sigma_omega, res_omega_mu1)
	plt.plot(tab_sigma_omega, res_omega_mu2)
	plt.xlabel("$\sigma_{\omega}$")
	plt.ylabel("moyene")
	plt.title("omega residuals")


def comparaison_minkowski(ds, N):
	"""
		makes a comparison for N different values of r factor in Minkowski formula error.
		uses the same generated dataset. 
		In : a Vehicle from Vehicle class in vehicle.py ; N : nb of points for the plot.
	"""

	res_V_sigma = np.zeros(N)
	res_V_mu = np.zeros(N)
	res_omega_sigma = np.zeros(N)
	res_omega_mu = np.zeros(N)

	r_minkowski = np.linspace(1,2,N)

	i=0

	for _r in r_minkowski:              #be careful : _r global variable.

		def minkowski_loss(y_true, y_pred):
			return K.mean(K.pow(K.abs(y_pred-y_true),_r))

		Rr_est, Rl_est, L_est = odo.all_param_AN(ds, minkowski_loss)

		residuals = odo.residuals(ds, Rr_est, Rl_est, L_est, False)
		res_V_sigma[i] = residuals[0][2]
		res_V_mu[i] = residuals[0][1]
		res_omega_sigma[i] = residuals[1][2]
		res_omega_mu[i] = residuals[1][1]

		i+=1


	plt.figure()

	plt.subplot(221)
	plt.plot(r_minkowski, res_V_sigma)
	plt.xlabel("$r_m$")
	plt.ylabel("ecart type")
	plt.title("V residuals")
	plt.subplot(222)

	plt.plot(r_minkowski, res_V_mu)
	plt.xlabel("$r_m$")
	plt.ylabel("moyenne")
	plt.title("V residuals")
	plt.subplot(223)

	plt.plot(r_minkowski, res_omega_sigma)
	plt.xlabel("$r_m$")
	plt.ylabel("ecart type")
	plt.title("omega residuals")
	plt.subplot(224)

	plt.plot(r_minkowski, res_omega_mu)
	plt.xlabel("$r_m$")
	plt.ylabel("moyene")
	plt.title("omega residuals")