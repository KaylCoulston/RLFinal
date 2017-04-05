import threading
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os
import gym

from time import time
#from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

class OneStepQWorker():
	#code
	def __init__(self, global_theta, global_theta_target, global_T):
		#global shared
		self.global_theta = global_theta
		self.global_theta_target = global_theta_target
		self.global_t = global_t
		self.global_max_t = 100 #TODO

		#local param
		self.local_t = 0
		self.local_target_weight = self.global_theta
		self.local_gradient = 0

		#TODO: init
		#self.Q = 
		#self.gamma =
			
		self.env = gym.make('Pong-v0')
		self.available_action = self.env.action_space.n
		

	def algorithimOne(self):
		
		#Get initial state
		state = self.env.reset()
		
		while(self.global_t < self.global_max_t):
			#Take action a with epsilon-greedy policy based on Q(s, a; theta)
			action = eGreedy(computeQ(state, action, self.global_theta))

			next_state, reward, terminal, _ = self.env.step(action)

			if(terminal):
				y = reward
			else:
				y = reward + self.gamma * np.argmax(computeQ(next_state, action, self.global_theta))
		
			#Accumulate gradients wrt theta
			self.local_gradient += accumulateGradient(y, q_last_value, self.global_theta, self.global_gradient)	

			state = next_state
			
			self.local_t += 1 
			self.global_t += 1 
			
			#update target
			if(self.global_t % self.i_target == 0):
				self.global_theta_target = self.global_theta
			
			if(self.local_t % self.i_async_update == 0 or terminal):
				asynchronousUpdate(self.local_gradient, self.global_gradient)
				
				#Clear self.local_gradient to 0		
				#TODO	



	def eGreedy(self, Q):
		#TODO
		action = 0 
		return action
				
	def getMaxQ(self, next_state, action, self.global_theta_target):
		#TODO
		q_value = 0
		return q_value

	def computeQ(self, state, action, theta):
		#TODO
		#self.Q[state, action, self.global_theta]
		return

	def accumulateGradient(y, q_last_value, self.global_theta, self.global_gradient):
		#TODO
		gradient = 0
		return gradient	


	def asynchronousUpdate(self.local_gradient, self.global_gradient):
		#TODO
		return
def main():
	#start

if name==__main__():
	main()
