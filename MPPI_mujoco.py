#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym
import numpy as np
import csv
import json


# In[9]:


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, gamma=0.99, log_file=None, noise_gaussian=True, downward_start=True):
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
        if self.env.unwrapped.spec.id == "Pendulum-v0":
            self.x_init = self.env.env.state
        elif self.env.unwrapped.spec.id == "HumanoidStandup-v2":
            self.x_init = env.sim.get_state()
        ############################
        
        self.gamma = gamma
        
        self.log_file = log_file
        if log_file is not None:
            self.writer = csv.writer(log_file, delimiter='\t')
            headers = ["timestamp", "reward", "action", "state"]
            self.writer.writerow(headers)
        

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, self.env.action_space.shape[0]))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)

    def _get_reward_from_state(self, s):
        root_z = s[0]
        if root_z > 1.1:
            return 1.0
        else:
            return 1.0 - (1.1 - root_z)
    
    def _compute_total_reward(self, k):
        discount = 1
        ############################
        if self.env.unwrapped.spec.id == "Pendulum-v0":
            self.env.env.state = self.x_init
        elif self.env.unwrapped.spec.id == "HumanoidStandup-v2":
            self.env.sim.set_state(self.x_init)
        ############################
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            s, reward, _, _ = self.env.step(np.array([perturbed_action_t]))
            if self.env.unwrapped.spec.id == "HumanoidStandup-v2":
                reward = self._get_reward_from_state(s)
            self.reward_total[k] += discount * reward
            discount *= self.gamma

    def _ensure_non_zero(self, reward, beta, factor):
        return np.exp(-factor * (beta - reward))


    def control(self, iter=1000):
        for timestamp in range(iter):
            for k in range(self.K):
                self._compute_total_reward(k)

            beta = np.max(self.reward_total)  # maximum reward of all trajectories
#             print()
#             print(self.reward_total)
#             print(beta)
            reward_total_non_zero = self._ensure_non_zero(reward=self.reward_total, beta=beta, factor=1/self.lambda_)
#             print(reward_total_non_zero)
            eta = np.sum(reward_total_non_zero)
            
            omega = 1/eta * reward_total_non_zero
#             print("Omega: {}".format(omega))
#             print("Noise: {}".format(self.noise))
#             print("U before: {}".format(self.U))
            self.U += [np.sum(omega.reshape(len(omega), 1) * self.noise[:, t], axis=0) for t in range(self.T)]
#             print("Incremental: {}".format([np.sum(omega.reshape(len(omega), 1) * self.noise[:, t], axis=0) for t in range(self.T)]))
#             print("U after: {}".format(self.U))
            ############################
            if self.env.unwrapped.spec.id == "Pendulum-v0":
                self.env.env.state = self.x_init
            elif self.env.unwrapped.spec.id == "HumanoidStandup-v2":
                self.env.sim.set_state(self.x_init)
            ############################
            s, r, _, _ = self.env.step(np.array([self.U[0]]))
            try:
                r = r[0]
            except:
                pass
            if self.env.unwrapped.spec.id == "HumanoidStandup-v2":
                r = self._get_reward_from_state(s)
            print("timestamp: {}, action taken: {} reward received: {}".format(timestamp, self.U[0], r))
            self.env.render()
#             self.env.sim.render(1024, 1024)

            self.U = np.roll(self.U, -1, axis=0)

            self.U[-1] = self.u_init  #
            self.reward_total[:] = 0
#             print("U after shifting: {}".format(self.U))
#             print("Rewards reset: {}".format(self.reward_total))
            
            ############################
            if self.env.unwrapped.spec.id == "Pendulum-v0":
                self.x_init = self.env.env.state
            elif self.env.unwrapped.spec.id == "HumanoidStandup-v2":
                self.x_init = env.sim.get_state()
            ############################
            
            if self.writer is not None:
                self._write_record(timestamp, r, self.U[0], s)
            
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, self.env.action_space.shape[0]))
    
    def _write_record(self, timestamp, reward, action, state):
        action_json = json.dumps(action.tolist())
        state_json = json.dumps(state.reshape(len(state), 1).tolist())
        self.writer.writerow([timestamp, reward, action_json, state_json])
        self.log_file.flush()


# In[10]:


ENV_NAME = "HumanoidStandup-v2"
TIMESTEPS = 8  # T
N_SAMPLES = 4  # K
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# TIMESTEPS = 15 # T
# N_SAMPLES = 120  # K

noise_mu = 0
noise_sigma = 0.2
lambda_ = 1.25
gamma = 0.99

env = gym.make(ENV_NAME)

from gym.wrappers import Monitor
env = Monitor(env, './video', force=True)
# env._max_episode_steps = 200
env.render()
# env.sim.render(1024, 1024)
print(env.observation_space)
print(env.action_space)

U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(TIMESTEPS, env.action_space.shape[0]))  # pendulum joint effort in (-2, +2)
# print(U)

log_file = open("train_record_tmp.tsv", "w")


mppi_gym = MPPI(env=env, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, gamma=gamma, log_file=log_file, noise_gaussian=True)
mppi_gym.control(iter=30)

log_file.close()


# In[ ]:




