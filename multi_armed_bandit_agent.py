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

	def update_mean_value_with_random_walk(self, mean=0, std=0.01):
		self.mean_value = self.mean_value + np.random.normal(mean, std, self.nb_arms)

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

	def update(self, action, reward, alpha=None):
		self.step += 1
		self.action_counter[action] += 1
		#if no parameter is specified, select sample averages action-value method
		if alpha == None :
			alpha = 1/self.action_counter[action]
		self.value_estimate[action] = self.value_estimate[action] \
			+ alpha*(reward - self.value_estimate[action])


# Train n_bandit_rl algorithms on a same agent
def train_bandits_rl(bandit_agent, epsilon, n_steps, n_bandits_rl, alpha=None):
	rwd = np.array(np.zeros((n_bandits_rl, n_steps)))
	bandit_rl = []
	act = np.array(np.zeros((n_bandits_rl, n_steps)), dtype = int)
	best_action = np.array(np.zeros((n_bandits_rl, n_steps)), dtype = int)
	bandit_agent_mean_value_start = bandit_agent.mean_value

	for i in range(0, n_bandits_rl):
		# Instanciate a learning algorithm
		bandit_rl.append(BanditRL(epsilon, n_steps))
		# Reinit the bandit agent mean value
		bandit_agent.mean_value = bandit_agent_mean_value_start

		# Init the learning algorithm for the first time steps corresponding
		# to the number of possible actions
		for j in range(0, bandit_rl[i].action_counter.size):
			act[i,j] = bandit_rl[i].select_action()
			best_action[i,j] = int(np.argmax(bandit_agent.mean_value))
			rwd[i,j] = bandit_agent.reward(act[i,j])
			bandit_rl[i].update(act[i,j], rwd[i,j], alpha)
			bandit_agent.update_mean_value_with_random_walk() # comment line for stationnary problem

		# Train the learning algorithm
		for j in range(bandit_rl[i].action_counter.size, n_steps):
			act[i,j] = bandit_rl[i].select_action()
			best_action[i,j] = int(np.argmax(bandit_agent.mean_value))
			rwd[i,j] = bandit_agent.reward(act[i,j])
			bandit_rl[i].update(act[i,j], rwd[i,j], alpha)
			bandit_agent.update_mean_value_with_random_walk() # comment line for stationnary proble

	avg_reward = np.average(rwd, 0)
	#print('avg_reward =', repr(avg_reward))

	correct_action = np.array(np.zeros(n_steps))
	#print('best_action =', repr(best_action))
	#print('----------------')
	for j in range(0, n_steps):
		#print('act[:,j]=', repr(act[:,j]))
		#print('best_action[j]=', repr(best_action[j]))
		correct_action[j] = np.count_nonzero(act[:,j] == best_action[:,j])/n_bandits_rl
		#print('correct_action[j]=', repr(correct_action[j]))

	return avg_reward, correct_action

# Instanciate an armed bandit with normally distributed reward value
bandit_agent = MultiArmedBandit(np.random.normal(0, 1, 10), np.array(np.ones(10)))
#bandit_agent = MultiArmedBandit(np.array([-3, -2, -1, 0, 1, 2, -1, -2, -3, -4]), \
#	np.array(np.ones(10)))
print("bandit mean value", repr(bandit_agent.mean_value))

n_steps = 10000
n_bandits_rl = 1000
epsilon = 0.1
avg_reward_sa, correct_action_sa \
	= train_bandits_rl(bandit_agent, epsilon, n_steps, n_bandits_rl)
avg_reward_cst, correct_action_cst \
	= train_bandits_rl(bandit_agent, epsilon, n_steps, n_bandits_rl, 0.1)

# Plot average reward and correct action
plt.figure()

# Reward
plt.subplot(211)
plt.plot(range(0, n_steps), avg_reward_sa, range(0, n_steps), avg_reward_cst, '--r', \
	[0,n_steps], [np.max(bandit_agent.mean_value), np.max(bandit_agent.mean_value)], '--k')
plt.ylabel("Average Reward")

# Correct action
plt.subplot(212)
plt.plot(range(0, n_steps), correct_action_sa, range(0, n_steps), correct_action_cst, '--r')
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
