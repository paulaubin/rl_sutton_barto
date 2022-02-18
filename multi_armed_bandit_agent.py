#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/02/17
# version ='1.0'
# header reference : https://www.delftstack.com/howto/python/common-header-python/
# ---------------------------------------------------------------------------
""" Sutton and Barto Multi-Armed Bandit Exercices """
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Define the MutliArmedBandit class
@dataclass
class MultiArmedBandit:
	"""An agent of a multi armed bandit"""
	nb_arms : int
	mean_value : np.array([])
	std_value  : np.array([])

	def __init__(self, mean_value=np.array([]), std_value=np.array([])):
		self.mean_value = np.array(np.zeros(10)) if mean_value.size==0 else mean_value
		self.std_value = np.array(np.ones(10)) if std_value.size==0 else std_value
		self.nb_arms = self.mean_value.size

	def reward(self, bandit_index):
		return np.random.normal(self.mean_value[bandit_index], self.std_value[bandit_index])

# Define the bandit RL algorithm
@dataclass
class BanditRL:
	"""A RL algorithm to solve the bandit problem"""
	step = 0
	action_counter = np.array([])
	value_estimate = np.array([])
	total_steps : int
	epsilon : float

	def __init__(self, epsilon=None, total_steps=None, value_estimate=np.array([])):
		self.epsilon = epsilon or 0.05
		self.total_steps = total_steps or 1000
		self.value_estimate = np.array(np.zeros(10)) \
							if value_estimate.size==0 else value_estimate
		self.action_counter = np.array(np.zeros(self.value_estimate.size))

	def select_action(self):
		exploitation_action = np.argmax(self.value_estimate)
		exploration_action = int(np.floor(np.random.uniform(0.0, 10.0)))
		if self.epsilon > np.random.uniform(0.0, 1.0):
			action = exploration_action
			#print("explore with action ", action)
		else:
			action = exploitation_action
			#print("exploit with action ", action)
		return action

	def update(self, action, reward):
		self.step += 1
		self.action_counter[action] += 1
		self.value_estimate[action] = self.value_estimate[action] \
			+ 1/self.action_counter[action]*(reward - self.value_estimate[action])

# Train n_bandit_rl algorithms on a same agent
def train_bandits_rl(bandit_agent, epsilon, n_steps, n_bandits_rl):
	rwd = np.array(np.zeros((n_bandits_rl, n_steps)))
	bandit_rl = []
	act = np.array(np.zeros((n_bandits_rl, n_steps)), dtype = int)

	for i in range(0, n_bandits_rl):
		# Instanciate a learning algorithm
		bandit_rl.append(BanditRL(epsilon, n_steps))

		# Init the learning algorithm
		for j in range(0, bandit_rl[i].value_estimate.size):
			act[i,j] = bandit_rl[i].select_action()
			rwd[i,j] = bandit_agent.reward(act[i,j])
			bandit_rl[i].update(act[i,j], rwd[i,j])

		# Train the learning algorithm
		for j in range(bandit_rl[i].value_estimate.size, n_steps):
			act[i,j] = bandit_rl[i].select_action()
			rwd[i,j] = bandit_agent.reward(act[i,j])
			bandit_rl[i].update(act[i,j], rwd[i,j])

	avg_reward = np.average(rwd, 0)
	#print('avg_reward =', repr(avg_reward))

	correct_action = np.array(np.zeros(n_steps))
	best_action = int(np.argmax(bandit_agent.mean_value))
	#print('best_action =', repr(best_action))
	for j in range(0, n_steps):
		#print('act[:,j]=', repr(act[:,j]))
		correct_action[j] = np.count_nonzero(act[:,j] == best_action)/act.shape[0]

	return avg_reward, correct_action

# Instanciate an armed bandit with normally distributed reward value
bandit_agent = MultiArmedBandit(np.random.normal(0, 1, 10), np.array(np.ones(10)))
#bandit_agent = MultiArmedBandit(np.array([-3, -2, -1, 0, 1, 2, -1, -2, -3, -4]), \
#	np.array(np.ones(10)))
print("bandit mean value", repr(bandit_agent.mean_value))

n_steps = 1000
n_bandits_rl = 2000
epsilon = 0.1
avg_reward_0p1, correct_action_0p1 \
	= train_bandits_rl(bandit_agent, epsilon, n_steps, n_bandits_rl)
epsilon = 0.01
avg_reward_0p01, correct_action_0p01 \
	= train_bandits_rl(bandit_agent, epsilon, n_steps, n_bandits_rl)

# Plot average reward and correct action
plt.figure()

# Reward
plt.subplot(211)
plt.plot(range(0, n_steps), avg_reward_0p1, range(0, n_steps), avg_reward_0p01, '--r' ,\
	[0,n_steps], [np.max(bandit_agent.mean_value), np.max(bandit_agent.mean_value)], '--k')
plt.ylabel("Average Reward")

# Correc action
plt.subplot(212)
plt.plot(range(0, n_steps), correct_action_0p1, range(0, n_steps), correct_action_0p01, '--r')
plt.ylabel('% Optimal Action')
plt.xlabel('Steps')
plt.show()

'''	
# Instanciate an armed bandit with normally distributed reward value
bandit_agent = MultiArmedBandit(np.random.normal(0, 1, 10), np.array(np.ones(10)))
#bandit_agent = MultiArmedBandit(np.array([-3, -2, -1, 0, 1, 2, -1, -2, -3, -4]), \
#	np.array(np.ones(10)))
print("bandit mean value", repr(bandit_agent.mean_value))

# Train the learning agent
n_steps = 1000;
n_bandits_rl = 200;
epsilon = 0.1
rwd = np.array(np.zeros((n_bandits_rl, n_steps)))
bandit_rl = []
act = np.array(np.zeros((n_bandits_rl, n_steps)), dtype = int)

for i in range(0, n_bandits_rl):
	# Instanciate a learning algorithm
	bandit_rl.append(BanditRL(epsilon, n_steps))

	# Init the learning algorithm
	for j in range(0, bandit_rl[i].value_estimate.size):
		act[i,j] = bandit_rl[i].select_action()
		rwd[i,j] = bandit_agent.reward(act[i,j])
		bandit_rl[i].update(act[i,j], rwd[i,j])

	# Train the learning algorithm
	for j in range(bandit_rl[i].value_estimate.size, n_steps):
		act[i,j] = bandit_rl[i].select_action()
		rwd[i,j] = bandit_agent.reward(act[i,j])
		bandit_rl[i].update(act[i,j], rwd[i,j])

avg_reward = np.average(rwd, 0)
#print('avg_reward =', repr(avg_reward))

correct_action = np.array(np.zeros(n_steps))
best_action = int(np.argmax(bandit_agent.mean_value))
print('best_action =', repr(best_action))
for j in range(0, n_steps):
	#print('act[:,j]=', repr(act[:,j]))
	correct_action[j] = np.count_nonzero(act[:,j] == best_action)/act.shape[0]
'''
