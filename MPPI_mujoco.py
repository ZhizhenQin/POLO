#!/usr/bin/env python
# coding: utf-8

# In[37]:


import gym
import numpy as np
import csv
import json


# In[48]:


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, gamma=0.99, writer=None, noise_gaussian=True, downward_start=True):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.reward_total = np.zeros(shape=(self.K))

        self.env = env
        self.env.reset()
        if downward_start:
            self.env.env.state = [np.pi, 1]
        ############################
#         self.x_init = self.env.env.state
        self.x_init = env.sim.get_state()
        ############################
        
        self.gamma = gamma
        self.writer = writer
        

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, env.action_space.shape[0]))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)

    def _compute_total_reward(self, k):
        discount = 1
        ############################
#         self.env.env.state = self.x_init
        self.env.sim.set_state(self.x_init)
        ############################
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            _, reward, _, _ = self.env.step(np.array([perturbed_action_t]))
            self.reward_total[k] += discount * reward
            discount *= self.gamma

    def _ensure_non_zero(self, reward, beta, factor):
        return np.exp(-factor * (beta - reward))


    def control(self, iter=1000):
        for timestamp in range(iter):
            for k in range(self.K):
                self._compute_total_reward(k)

            beta = np.max(self.reward_total)  # maximum reward of all trajectories
            reward_total_non_zero = self._ensure_non_zero(reward=self.reward_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(reward_total_non_zero)
            
            omega = 1/eta * reward_total_non_zero

            
            self.U += [np.sum(omega.reshape(len(omega), 1) * self.noise[:, t], axis=0) for t in range(self.T)]

            ############################
#             self.env.env.state = self.x_init
            self.env.sim.set_state(self.x_init)
            ############################
            s, r, _, _ = self.env.step(np.array([self.U[0]]))
            print("action taken: {} reward received: {}".format(self.U[0], r))
            self.env.render()

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.reward_total[:] = 0
            
            ############################
#             self.x_init = self.env.env.state
            self.x_init = env.sim.get_state()
            ############################
            
            if self.writer is not None:
                self._write_record(timestamp, r, self.U[0], s)
    
    def _write_record(self, timestamp, reward, action, state):
        action_json = json.dumps(list(action))
        state_json = json.dumps(list(state))
        self.writer.writerow([timestamp, reward, action_json, state_json])


# In[50]:


ENV_NAME = "HumanoidStandup-v2"
TIMESTEPS = 64  # T
N_SAMPLES = 120  # K
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

noise_mu = 0
noise_sigma = 0.2
lambda_ = 1.25
gamma = 0.99

env = gym.make(ENV_NAME)
# env.render()
print(env.observation_space)
print(env.action_space)

U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(TIMESTEPS, env.action_space.shape[0]))  # pendulum joint effort in (-2, +2)
# print(U)

log_file = open("train_record.tsv", "w")
writer = csv.writer(log_file, delimiter='\t')
headers = ["timestamp", "reward", "action", "state"]
writer.writerow(headers)

mppi_gym = MPPI(env=env, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, gamma=gamma, writer=writer, noise_gaussian=True)
mppi_gym.control(iter=10000)

log_file.close()

