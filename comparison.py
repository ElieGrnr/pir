#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os, logging, numpy as np, matplotlib.pyplot as plt
import keras, pickle
import keras.backend as K
import pdb
import odometry_test as od
from vehicle import Vehicle

def comparison_noise(sigma_V, sigma_omega):
	"""
		makes a comparison of relative error  according 3 methods
		Method 1 : pseudoinv
		Method 2 : 2 AN
		Method 3 : single AN
		In : sigma_V : noise for V
			 sigma_omega : noise for omega
		Out : an array([[error_Rr_method1, error_Rl_method1,error_L_method1],
						[error_Rr_method2, error_Rl_method2,error_L_method2],
						[error_Rr_method3, error_Rl_method3,error_L_method3]])
	"""
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

def plot_comparison_noise(sigma_V_max, sigma_omega_max):
	"""
		Makes a comparison of relative error according 3 methods and according noise
		Method 1 : pseudoinv
		Method 2 : 2 AN
		Method 3 : single AN
		In : sigma_V : noise from 0 to sigma_V
			 sigma_omega : noise from 0 to sigma_omega
		Out : a plot
	"""
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


def comparaison_minkowski(Robot, N):
	"""
		makes a comparison for N different values of r factor in Minkowski formula error.
		uses the same generated dataset. 
		In : a Vehicle from Vehicle class in vehicle.py ; N : nb of points for the plot.
	"""
	ds = Robot.generate_random_data(-10, 10, 5000, True, 0.5, 0.5, 1.5, 1.5)
	Rr = Robot.right_wheel_radius 		#true Rr
	Rl = Robot.left_wheel_radius  		#true Rl
	L = Robot.space_wheels 				#true L
	relative_error = []
	r_minkowski = np.linspace(1,2,N)
	for _r in r_minkowski:              #be careful : _r global variable.

		def minkowski_loss(y_true, y_pred):
			return K.mean(K.pow(K.abs(y_pred-y_true),_r))

		Rr_est, Rl_est, L_est = od.all_param_AN(ds, minkowski_loss)

		relative_error.append([100*abs(Rr_est-Rr)/Rr, 100*abs(Rl_est-Rl)/Rl, 100*abs(L_est-L)/L])
		#relative error = N*3 array

	plt.figure()
	plt.subplot(311)
	plt.plot(r_minkowski, [relative_error[i][0] for i in range(N)])
	plt.xlabel("r")
	plt.ylabel("erreur relative")
	plt.title("roue droite")
	plt.subplot(312)
	plt.plot(r_minkowski, [relative_error[i][1] for i in range(N)])
	plt.xlabel("r")
	plt.ylabel("erreur relative")
	plt.title("roue gauche")
	plt.subplot(313)
	plt.plot(r_minkowski, [relative_error[i][2] for i in range(N)])
	plt.xlabel("r")
	plt.ylabel("erreur relative")
	plt.title("espace")
	plt.show()
	


Robot = Vehicle(0.08, 0.081, 0.2, 0.1, 0.5)
comparaison_minkowski(Robot, 8)

#plot_comparison(0.5, 0.5)





