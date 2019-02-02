#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt
import random
import keras, pickle
import pdb

class Vehicle:
    def __init__(self, Rr, Rl, space_wheels):
        self.right_wheel_radius = Rr
        self.left_wheel_radius = Rl
        self.space_wheels = space_wheels

    def generate_random_data(self, w_min, w_max, Nsample, outliers=False, ratio_V=0.1, ratio_omega=0.1, coef_V=1, coef_omega=1):
        Rr = self.right_wheel_radius
        Rl = self.left_wheel_radius
        L = self.space_wheels
        wr = np.random.uniform(w_min, w_max, Nsample)
        wl = np.random.uniform(w_min, w_max, Nsample)
        V = 0.5*(Rr*wr+Rl*wl)+0.01*np.random.randn(Nsample)
        omega = 0.5*(Rr*wr-Rl*(wl))/L+0.01*np.random.randn(Nsample)

        if outliers:
            nb_outliers_V = int(ratio_V*Nsample)
            nb_outliers_omega = int(ratio_omega*Nsample)
            index = [i for i in range(Nsample)]
            index_V = random.sample(index, nb_outliers_V)
            index_omega = random.sample(index, nb_outliers_omega)
            for i in index_V:
                V[i] += coef_V*random.uniform(0.5*(Rr*w_max+Rl*w_max),0.5*(Rr*w_min+Rl*w_min))
            for i in index_omega:
                omega[i] += coef_omega*random.uniform(0.5*(Rr*w_max-Rl*w_min)/L,0.5*(Rr*w_min-Rl*w_max)/L)

        ds = np.array([wr, wl, V, omega])
        return ds

