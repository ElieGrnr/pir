#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt
import random
import keras, pickle
import pdb

class Vehicle:
    """
        A vehicle is:
        Right wheel radius
        left wheel radius
        space beetwen wheels
        noise (sigma) for V
        noise (sigma) for omega
    """
    def __init__(self, Rr, Rl, space_wheels, sigma_V, sigma_omega):
        self.right_wheel_radius = Rr
        self.left_wheel_radius = Rl
        self.space_wheels = space_wheels
        self.sigma_V = sigma_V
        self.sigma_omega = sigma_omega

    def generate_random_data(self, w_min, w_max, Nsample, outliers=False, ratio_V=0.1, ratio_omega=0.1, coef_V=1, coef_omega=1):
        """
            Generate random data set.
            In :
            Uniform distribution of wr and wl between w_lin and w_max
            Nsample : numbre of sample
            outliers : True or False : uniform inside the wr and wl lists.
            ratio_V : ratio outliers for V
            ratio_omega : ratio outliers for omega
            coef_V : force of the outliers for V
            coef_omega : force of the outliers for omega
            Out:
            dataset = np.array([wr, wl, V, omega])
        """
        Rr = self.right_wheel_radius
        Rl = self.left_wheel_radius
        L = self.space_wheels
        sigma_V = self.sigma_V
        sigma_omega = self.sigma_omega
        wr = np.random.uniform(w_min, w_max, Nsample) #loi uniforme
        wl = np.random.uniform(w_min, w_max, Nsample)
        V = 0.5*(Rr*wr+Rl*wl)+sigma_V*np.random.randn(Nsample) #calcul de V avec bruit
        omega = 0.5*(Rr*wr-Rl*(wl))/L+sigma_omega*np.random.randn(Nsample)

        if outliers:
            nb_outliers_V = int(ratio_V*Nsample)
            nb_outliers_omega = int(ratio_omega*Nsample)
            index = [i for i in range(Nsample)]
            index_V = random.sample(index, nb_outliers_V) #échantillon des indices des outliers (aléa)
            index_omega = random.sample(index, nb_outliers_omega)
            for i in index_V:
                V[i] += coef_V*random.uniform(0.5*(Rr*w_max+Rl*w_max),0.5*(Rr*w_min+Rl*w_min)) #V + coef_V*
                #uniform(Vmin, Vmax)
            for i in index_omega:
                omega[i] += coef_omega*random.uniform(0.5*(Rr*w_max-Rl*w_min)/L,0.5*(Rr*w_min-Rl*w_max)/L)

        ds = np.array([wr, wl, V, omega])
        return ds

