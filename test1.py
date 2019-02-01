#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import pi
import matplotlib.pyplot as plt

Nechantillion = 30

Rl = 8
Rr = 8.1
L = 20

om_min, om_max = -1, 1
#wr = np.linspace(0,10*2*pi,Nechantillion)+np.random.uniform(0,1,Nechantillion)
wr = np.random.uniform(om_min, om_max, Nechantillion)
#wl = np.linspace(0,10*2*pi,Nechantillion)+np.random.uniform(0,1,Nechantillion)
wl = np.random.uniform(om_min, om_max, Nechantillion)

V = 0.5*(Rr*wr+Rl*wl)+0.1*np.random.randn(Nechantillion)
omega = 0.5*(Rr*wr-Rl*(-wl))/L+0.1*np.random.randn(Nechantillion)

plt.subplot(1, 4, 1)
plt.hist(wr)
plt.subplot(1, 4, 2)
plt.hist(wl)
plt.subplot(1, 4, 3)
plt.hist(V)
plt.subplot(1, 4, 4)
plt.hist(omega)
plt.show()



H1 = np.zeros((Nechantillion,2))
H1[:,0] = wr*0.5
H1[:,1] = wl*0.5

X1 = np.dot(np.linalg.pinv(H1),V)

Rl_est, Rr_est = X1[0], X1[1]

H2 = np.zeros((Nechantillion,1))
H2[:,0] = 0.5*(Rr_est*wr-Rl_est*(-wl))

X2 = np.dot(np.linalg.pinv(H2),omega)

L_est = 1/X2[0]

print("Rayon roue droite estimé : ", Rr_est)
print("Rayon roue gauche estimé : ", Rl_est)
print("Voie estimée : ", L_est)
