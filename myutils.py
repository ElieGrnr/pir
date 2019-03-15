#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from math import sqrt
import numpy as np

import homere_control.io_dataset as iodata


def data_converter(filename, type):
	"""
		Converts Drouin's data test and python classes properties into a datset compliant with my functions.
		filename :  '/home/poine/work/homere/homere_control/data/homere_io_10.npz'
		type : 'homere'

		Out : ds = [wr, wl, V, omega]
	"""

	original_ds = iodata.DataSet(filename, type)

	wl = original_ds.enc_vel_lw
	wr = original_ds.enc_vel_rw

	original_ds.truth_lvel_body = iodata.interpolate(original_ds.truth_lvel_body, original_ds.truth_vel_stamp, original_ds.enc_vel_stamp)
	original_ds.truth_rvel = iodata.interpolate(original_ds.truth_rvel, original_ds.truth_vel_stamp, original_ds.enc_vel_stamp)

	truth_vx, truth_vy = original_ds.truth_lvel_body[:,0], original_ds.truth_lvel_body[:,1]
	truth_V = [sqrt(truth_vx[i]**2+truth_vy[i]**2) for i in range(len(truth_vy))]
	truth_omega = original_ds.truth_rvel[:,2]

	print len(wr), len(wl), len(truth_omega), len(truth_V)

	time_encoders = original_ds.enc_vel_stamp
	#time_velocity = original_ds.truth_vel_stamp	

	converted_ds = [wr, wl, truth_V, truth_omega]

	return converted_ds, time_encoders

#filename, type = './data/oscar_io_oval.npz', 'oscar'
#ds, time = data_converter(filename, type)





