#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt
import keras, pickle
import pdb
import odometry_test as od
from vehicle import Vehicle

def comparison_noise(sigma_V, sigma_omega):
	Robot = Vehicle(0.08, 0.081, 0.2, sigma_V, sigma_omega)
	Rr = Robot.right_wheel_radius
	Rl = Robot.left_wheel_radius
	L = Robot.space_wheels
	ds = Robot.generate_random_data(-10, 10, 5000, False)

	Rr_est3, Rl_est3, L_est3 = od.all_param_AN(ds)

	Rr_est2, Rl_est2 = od.wheels_radius_AN(ds)
	L_est2 = od.space_wheels_AN(ds, (Rr_est2, Rl_est2))

	Rr_est1, Rl_est1 = od.wheels_radius_INV(ds)
	L_est1 = od.space_wheels_INV(ds, (Rr_est1, Rl_est1))

	error = np.array([[(Rr_est1-Rr)/Rr, (Rl_est1-Rl)/Rl, (L_est1-L)/L], 
		[(Rr_est2-Rr)/Rr, (Rl_est2-Rl)/Rl, (L_est2-L)/L],
		[(Rr_est3-Rr)/Rr, (Rl_est3-Rl)/Rl, (L_est3-L)/L]])
	return error

def plot_comparison(sigma_V_max, sigma_omega_max):
	tab_sigma_V = np.linspace(0, sigma_V_max, 3)
	tab_sigma_omega = np.linspace(0, sigma_omega_max, 3)
	error_V = []
	error_omega = []
	for i in range(len(tab_sigma_V)):
		error_V.append(comparison_noise(tab_sigma_V[i], 0.01))
	for i in range(len(tab_sigma_omega)):
		error_omega.append(comparison_noise(0.01, tab_sigma_omega[i]))

	print(np.shape(error_V))
	print(np.shape(error_omega))


	plt.figure()

	plt.subplot(321, xlabel="noise $\sigma_V$", ylabel="error")
	plt.title("erreur relative de $R_r$")
	plt.plot(tab_sigma_V, error_V[:][:][0], label=["méthode 1", "méthode 2", "méthode 3"])
	plt.legend()

	plt.subplot(323, xlabel="noise $\sigma_V$", ylabel="error")
	plt.title("erreur relative de $R_l$")
	plt.plot(tab_sigma_V, error_V[:][:][1], label=["méthode 1", "méthode 2", "méthode 3"])
	plt.legend()

	plt.subplot(325, xlabel="noise $\sigma_V$", ylabel="error")
	plt.title("erreur relative de $L$")
	plt.plot(tab_sigma_V, error_V[:][:][2], label=["méthode 1", "méthode 2", "méthode 3"])
	plt.legend()

	plt.subplot(321, xlabel="noise $\sigma_\omega$", ylabel="error")
	plt.title("erreur relative de $R_r$")
	plt.plot(tab_sigma_omega, error_omega[:][:][0], label=["méthode 1", "méthode 2", "méthode 3"])
	plt.legend()

	plt.subplot(323, xlabel="noise $\sigma_\omeg$", ylabel="error")
	plt.title("erreur relative de $R_l$")
	plt.plot(tab_sigma_omega, error_omega[:][:][1], label=["méthode 1", "méthode 2", "méthode 3"])
	plt.legend()

	plt.subplot(325, xlabel="noise $\sigma_\omega$", ylabel="error")
	plt.title("erreur relative de $L$")
	plt.plot(tab_sigma_omega, error_omega[:][:][2], label=["méthode 1", "méthode 2", "méthode 3"])
	plt.legend()

	plt.show()

plot_comparison(0.5, 0.5)





