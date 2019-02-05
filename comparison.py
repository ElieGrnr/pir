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

	error = np.array([[abs((Rr_est1-Rr)/Rr), abs((Rl_est1-Rl)/Rl), abs((L_est1-L)/L)], 
		[abs((Rr_est2-Rr)/Rr), abs((Rl_est2-Rl)/Rl), abs((L_est2-L)/L)],
		[abs((Rr_est3-Rr)/Rr), abs((Rl_est3-Rl)/Rl), abs((L_est3-L)/L)]])
	return error

def plot_comparison(sigma_V_max, sigma_omega_max):
	tab_sigma_V = np.linspace(0, sigma_V_max, 5)
	tab_sigma_omega = np.linspace(0, sigma_omega_max, 5)
	error_V = []
	error_omega = []
	for i in range(len(tab_sigma_V)):
		error_V.append(comparison_noise(tab_sigma_V[i], 0.01))
	for i in range(len(tab_sigma_omega)):
		error_omega.append(comparison_noise(0.01, tab_sigma_omega[i]))

	print(np.shape(error_V))
	print(np.shape(error_omega))
	print(np.shape(error_V[:][:][0]))


	plt.figure()

	plt.subplot(321, xlabel="noise sigma_V", ylabel="error")
	plt.title("erreur relative de R_r")
	y1 = [error_V[i][0][0] for i in range(5)]
	y2 = [error_V[i][1][0] for i in range(5)]
	y3 = [error_V[i][2][0] for i in range(5)]
	plt.plot(tab_sigma_V, y1,label="methode1")
	plt.plot(tab_sigma_V, y2,label="methode2")
	plt.plot(tab_sigma_V, y3,label="methode3")
	plt.legend()

	plt.subplot(323, xlabel="noise sigma_V", ylabel="error")
	plt.title("erreur relative de R_l")
	y1 = [error_V[i][0][1] for i in range(5)]
	y2 = [error_V[i][1][1] for i in range(5)]
	y3 = [error_V[i][2][1] for i in range(5)]
	plt.plot(tab_sigma_V, y1)
	plt.plot(tab_sigma_V, y2)
	plt.plot(tab_sigma_V, y3)

	plt.subplot(325, xlabel="noise sigma_V", ylabel="error")
	plt.title("erreur relative de L")
	y1 = [error_V[i][0][2] for i in range(5)]
	y2 = [error_V[i][1][2] for i in range(5)]
	y3 = [error_V[i][2][2] for i in range(5)]
	plt.plot(tab_sigma_V, y1)
	plt.plot(tab_sigma_V, y2)
	plt.plot(tab_sigma_V, y3)

	plt.subplot(322, xlabel="noise sigma_omega", ylabel="error")
	plt.title("erreur relative de R_r")
	y1 = [error_omega[i][0][0] for i in range(5)]
	y2 = [error_omega[i][1][0] for i in range(5)]
	y3 = [error_omega[i][2][0] for i in range(5)]
	plt.plot(tab_sigma_omega, y1)
	plt.plot(tab_sigma_omega, y2)
	plt.plot(tab_sigma_omega, y3)

	plt.subplot(324, xlabel="noise sigma_omega", ylabel="error")
	plt.title("erreur relative de R_l")
	y1 = [error_omega[i][0][1] for i in range(5)]
	y2 = [error_omega[i][1][1] for i in range(5)]
	y3 = [error_omega[i][2][1] for i in range(5)]
	plt.plot(tab_sigma_omega, y1)
	plt.plot(tab_sigma_omega, y2)
	plt.plot(tab_sigma_omega, y3)

	plt.subplot(326, xlabel="noise sigma_omega", ylabel="error")
	plt.title("erreur relative de L")
	y1 = [error_omega[i][0][2] for i in range(5)]
	y2 = [error_omega[i][1][2] for i in range(5)]
	y3 = [error_omega[i][2][2] for i in range(5)]
	plt.plot(tab_sigma_omega, y1)
	plt.plot(tab_sigma_omega, y2)
	plt.plot(tab_sigma_omega, y3)

	plt.show()

plot_comparison(0.5, 0.5)





