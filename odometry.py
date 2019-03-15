from __future__ import division

from vehicle import Vehicle
import myutils as ut

from math import cos, sin 
import matplotlib.pyplot as plt

class State:

	def __init__(self, x=0, y=0, theta=0, v=0, omega=0):
		self.x = x
		self.y = y
		self.v = v
		self.theta = theta
		self.omega = omega

	def update(self, v, omega, dt):
		self.x = self.x + v*cos(self.theta)*dt
		self.y = self.y + v*sin(self.theta)*dt
		self.theta = self.theta + omega * dt
		self.v = v
		self.omega = omega


class LinearOdometer:

	def __init__(self, ds, time, linear_formula):
		self.wr = ds[0]
		self.wl = ds[1]
		self.time = time
		self.linear_formula = linear_formula # =[a, b, c, d]
		self.index = 0

	def compute(self):
		i = self.index
		[a, b, c, d] = self.linear_formula
		v = a*self.wr[i] + b*self.wl[i]
		omega = c*self.wr[i] + d*self.wl[i]
		if i==0:
			dt = self.time[1]-self.time[0]
		else:
			dt = self.time[i]-self.time[i-1]
		self.index += 1
		return v, omega, dt


def compute_position(ds, time, linear_formula):
	x = []
	y = []
	theta = []
	Nsample = len(time)

	state = State()
	odometer = LinearOdometer(ds, time, linear_formula)

	x.append(state.x)
	y.append(state.y)
	theta.append(state.theta)

	for i in range(Nsample):
		v, omega, dt = odometer.compute()
		state.update(v, omega, dt)
		x.append(state.x)
		y.append(state.y)
		theta.append(state.theta)

	return x, y, theta



filename, type = './data/oscar_io_oval.npz', 'oscar'
ds, time = ut.data_converter(filename, type)

x, y, theta = compute_position(ds, time, [1, 1, 1, 1])

plt.figure()
plt.plot(x, y)
plt.show()














	



