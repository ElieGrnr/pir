#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
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


    def generate_data(self, w_min, w_max, Nsample):
        """
            Generate random data set.
            In :
            Uniform distribution of wr and wl between w_min and w_max
            Nsample : numbre of sample
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
        omega = (Rr*wr-Rl*(wl))/L+sigma_omega*np.random.randn(Nsample) #*0.5 ???????????
        ds = np.array([wr, wl, V, omega])
        return ds

    def generate_outliers_uniform(self, w_min, w_max, Nsample, ratio_V=0.1, ratio_omega=0.1, bias_V=0, bias_omega=0, coef_V=1, coef_omega=1):
        """
            Generates outliers of a dataset [wr, wl, V, omega]. Uniform distribution of ouliers in ds. 
            Returns ds
        """
        Rr = self.right_wheel_radius
        Rl = self.left_wheel_radius
        L = self.space_wheels

        ds = self.generate_data(w_min, w_max, Nsample)

        nb_outliers_V = int(ratio_V*Nsample)
        nb_outliers_omega = int(ratio_omega*Nsample)
        index = [i for i in range(Nsample)]
        index_V = random.sample(index, nb_outliers_V) #échantillon des indices des outliers (aléa)
        index_omega = random.sample(index, nb_outliers_omega)
        for i in index_V:
            ds[2][i] += bias_V+coef_V*random.uniform(0.5*(Rr*w_max+Rl*w_max),0.5*(Rr*w_min+Rl*w_min)) #V + coef_V*
            #uniform(Vmin, Vmax)
        for i in index_omega:
            ds[3][i] += bias_omega+coef_omega*random.uniform(0.5*(Rr*w_max-Rl*w_min)/L,0.5*(Rr*w_min-Rl*w_max)/L)
        return ds


