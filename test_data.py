#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt
import pdb
#from vehicle import Vehicle
#import description as descr

import homere_control.io_dataset as iodata

filename, _type = '/home/poine/work/julie/julie/julie_control/scripts/julie_odom_data_1.npz', 'homere'
ds = iodata.DataSet(filename, _type)
iodata.plot2d(ds)

"""for i in range(1, 11):
	filename, _type = '/home/poine/work/homere/homere_control/data/homere_io_{}.npz'.format(i), 'homere'
	ds = iodata.DataSet(filename, _type)
	iodata.plot2d(ds)
	plt.title("test {}".format(i))
#iodata.plot_all(_ds)"""
plt.show()