#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

import homere_control.io_dataset as iodata


def data_converter(filename, type):
	"""
		Converts Drouin's data test and python classes properties into a datset compliant with my functions.
		filename :  '/home/poine/work/homere/homere_control/data/homere_io_10.npz'
		type : 'homere'

		Out : ds = [wr, wl, V, omega], time (list of instant)
		len(wr) = len(time)
	"""

	original_ds = iodata.DataSet(filename, type)

	wl = original_ds.enc_vel_lw
	wr = original_ds.enc_vel_rw

	original_ds.truth_lvel_body = iodata.interpolate(original_ds.truth_lvel_body, original_ds.truth_vel_stamp, original_ds.enc_vel_stamp)
	original_ds.truth_rvel = iodata.interpolate(original_ds.truth_rvel, original_ds.truth_vel_stamp, original_ds.enc_vel_stamp)

	truth_vx, truth_vy = original_ds.truth_lvel_body[:,0], original_ds.truth_lvel_body[:,1]
	truth_V = [sqrt(truth_vx[i]**2+truth_vy[i]**2) for i in range(len(truth_vy))]
	truth_omega = original_ds.truth_rvel[:,2]

	time_encoders = original_ds.enc_vel_stamp	

	converted_ds = [wr, wl, truth_V, truth_omega]

	return converted_ds, time_encoders

def data_initial_position(filename, type):
	original_ds = iodata.DataSet(filename, type)
	x0, y0, theta0 = original_ds.truth_pos[0][0], original_ds.truth_pos[0][1], original_ds.truth_yaw[0]
	return x0, y0, theta0

def plot_truth(filename, type):
	original_ds = iodata.DataSet(filename, type)
	plt.plot(original_ds.truth_pos[:,0], original_ds.truth_pos[:,1], label='mocap')
	plt.axis('equal')
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')


#filename, type = './data/oscar_io_oval.npz', 'oscar'
#ds, time = data_converter(filename, type)