#!/usr/bin/env python
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt

#import odom_dataset as odm_ds
import homere_control.io_dataset

ds = homere_control.io_dataset.DataSet('/home/poine/work/homere/homere_control/data/homere_io_2.npz', _type='homere')
homere_control.io_dataset.plot_encoders(ds)
plt.show()
