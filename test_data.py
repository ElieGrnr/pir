#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt
import pdb
#from vehicle import Vehicle
#import description as descr

import homere_control.io_dataset as iodata

filename, _type = '/home/poine/work/homere/homere_control/data/homere_io_10.npz', 'homere'
_ds = iodata.DataSet(filename, _type)
pdb.set_trace()
iodata.plot_all(_ds)
plt.show()