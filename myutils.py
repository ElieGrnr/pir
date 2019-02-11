#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import homere_control.io_dataset as iodata
import keras.backend as K 

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
	truth_lvel_body_1 = iodata.interpolate(original_ds.truth_lvel_body, original_ds.truth_vel_stamp, original_ds.enc_vel_stamp)
	truth_rvel_1 = iodata.interpolate(original_ds.truth_rvel, original_ds.truth_vel_stamp,original_ds.enc_vel_stamp)
	V = truth_lvel_body_1[:,0]
	omega = truth_rvel_1[:,2]
	converted_ds = [wr, wl, V, omega]
	return converted_ds


def minkowski_loss(y_true, y_pred):
	"""
		Compute Minkowski's loss for AN. Better than MSE if outliers.
	"""
	r = 0.4 #often 0.4
	return r*K.sum(K.pow(K.abs(y_pred-y_true)),r) 


