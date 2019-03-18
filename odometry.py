from __future__ import division

from vehicle import Vehicle
import myutils as ut
import odometry_test as odo

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


class LinearOdometer(State):

	def __init__(self, ds, time, linear_formula):
		self.wr = ds[0]
		self.wl = ds[1]
		self.time = time #len(time)=len(wr)=len(wl)
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


	def compute_position(self, initial_position):
		self.x_positions = [] # a list of x_position
		self.y_positions = [] # a list of y_position
		self.theta_positions = [] # a list of theta_position
		Nsample = len(self.time)

		x0, y0, theta0 = initial_position[0], initial_position[1], initial_position[2],
		self.x, self.y, self.theta = x0, y0, theta0

		self.x_positions.append(x0)
		self.y_positions.append(y0)
		self.theta_positions.append(theta0)

		for i in range(Nsample):
			v, omega, dt = self.compute()
			self.update(v, omega, dt)
			self.x_positions.append(self.x)
			self.y_positions.append(self.y)
			self.theta_positions.append(self.theta)

		return self.x_positions, self.y_positions, self.theta_positions #list

	def plot(self, label):
		plt.axis('equal')
		plt.grid('on', 'both', 'both', linestyle='--')
		plt.plot(self.x_positions, self.y_positions, label=label)


















	



