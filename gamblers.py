#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/03/01
# version ='1.0'
# header reference : https://www.delftstack.com/howto/python/common-header-python/
# ---------------------------------------------------------------------------
""" Sutton and Barto Gambler's Problem """
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Define the MutliArmedBandit class
@dataclass
class coin:
	"""A coin"""
	p_heads : float

	def __init__(self, p_heads=None):
		self.p_heads = 0.5 if p_heads==None else p_heads

	def toss(self):
		outcome = heads;
		if np.random.unfiform(0.0,1.0) > p_heads:
			outcome = tails
		else:
			outcome = heads
		return outcome

# Define the agent
@dataclass
class gambler:
	"""A gambler looking to gain 100€"""
	#capital = range(0,101) # From 0 to 100, 0 and 100 being the final states
	value_estimate = np.array(np.zeros(101))
	value_estimate[100] = 1.0
	policy = np.array(np.zeros(101))
	theta = 1e-2
	value_iteration_counter = 0
	p_heads : float
	capital : int

	def __init__(self, p_heads=None, capital=None):
		self.p_heads = 0.5 if p_heads==None else p_heads
		self.capital = 50 if capital==None else capital

	def update_per_capital(self, capital):
		stake_range = np.arange(1,1 + min(capital, 100-capital))
		max_action_value = np.zeros(len(stake_range))
		for s in stake_range:
			max_action_value[s-1] = self.p_heads*(0.0 + self.value_estimate[capital + s]) \
						+ (1 - self.p_heads)*(0.0 + self.value_estimate[capital - s])
		self.value_estimate[capital] = np.max(max_action_value)
		self.policy[capital] = np.argmax(max_action_value)+1

'''
	def update(self):
		delta = 0.0
		value_iteration_counter = 0
		while delta < theta: # check that the loop persists
			value_iteration_counter += 1
			v = self.value_estimate
			for c in rante(1,100):
				self.value_update_per_capital(c)
			delta = max(delta, np.linalg.norm(v - self.value_estimate))
'''

# Train and print metrics of the agent
g = gambler(0.4)
print("gambler is = ", repr(g))
value_estimate_log = np.array(g.value_estimate)
policy_log = np.array(g.policy)
delta = np.inf
iteration_counter = 0
theta = 1e-4
while delta > theta :
#for i in range(1000):
	iteration_counter += 1
	v = np.array(g.value_estimate)
	for c in range(1,100):
		g.update_per_capital(c)
	value_estimate_log = np.vstack((value_estimate_log, np.array(g.value_estimate)))
	policy_log = np.vstack((policy_log, np.array(g.policy)))
	delta = np.linalg.norm(v - g.value_estimate)
	#print("v=", repr(v))
	#print("g.value_estimate=", repr(g.value_estimate))
	#print("np.linalg.norm(v - g.value_estimate)=", repr(np.linalg.norm(v - g.value_estimate)))
	print("------------------------------------------------------")	
	print("step = ", iteration_counter)
	print("delta = ", delta)
	#if iteration_counter % 10 == 0:
print("value_estimate_log=", repr(value_estimate_log[-1]))
print("policy_log=", repr(policy_log[-1]))