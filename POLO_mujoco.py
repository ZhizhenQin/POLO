#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import csv
import json


# In[2]:


import torch
import torch.nn as nn
from scipy.special import softmax


# In[ ]:


RENDER = False


# In[3]:


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(ValueNet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )
    
    def forward(self, x):
        return self.model(x)


# In[4]:


class MLP(nn.Module):
    def __init__(self,
                 input_dim : int,
                 hidden : int,
                 output_dim : int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, output_dim)
        
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)
        
    def forward(self, inputs):
        x = self.l1(inputs)
        x = nn.functional.tanh(x)
        x = self.l2(x)
        x = nn.functional.tanh(x)
        x = self.l3(x)
        return x


# In[5]:


class ModelWithPrior(nn.Module):
    def __init__(self,
                 base_model : nn.Module,
                 prior_model : nn.Module,
                 prior_scale : float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.prior_model = prior_model
        self.prior_scale = prior_scale
        
    def forward(self, inputs):
        with torch.no_grad():
            prior_out = self.prior_model(inputs)
            prior_out = prior_out.detach()
        model_out = self.base_model(inputs)
        return model_out + (self.prior_scale * prior_out)


# In[6]:


class POLO(object):
    def __init__(self, env, K, T, U, lambda_, noise_mu, 
                 noise_sigma, u_init, memory_size, observation_space, action_space, state_space, 
                 net_hidden_layers, num_nets, state_samples, gradient_steps, gamma=0.99, log_file=None, 
                 noise_gaussian=True):
        
        self.memory_size = memory_size
        self.obs_mem = np.zeros((self.memory_size, observation_space))
        self.state_mem = [None for i in range(self.memory_size)]
        self.targets_mem = np.zeros((self.memory_size, num_nets))
        
        self.num_nets = num_nets
        
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.reward_total = np.zeros(shape=(self.K))
        
        self.state_samples = state_samples
        self.gradient_steps = gradient_steps

        self.env = env

        ############################
        if self.env.unwrapped.spec.id == "Pendulum-v0":
            self.x_init = self.env.env.state
        elif self.env.unwrapped.spec.id == "HumanoidStandup-v2":
            self.x_init = env.sim.get_state()
        ############################
        
        self.gamma = gamma
        
        self.max_reward_for_net = np.full((self.num_nets), '-inf', dtype=np.float)
        
        self.log_file = log_file
        if log_file is not None:
            self.writer = csv.writer(log_file, delimiter='\t')
            headers = ["timestamp", "reward", "action", "state"]
            self.writer.writerow(headers)
        

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, self.env.action_space.shape[0]))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)
            
        self._build_value_nets(observation_space, net_hidden_layers, 1)
        
    def _build_value_nets(self, input_dim, hidden, output_dim):
        self.value_nets = []
        self.loss_funcs = []
        self.optimizers = []
        
        for i in range(self.num_nets):
#             self.value_nets.append(ValueNet(input_dim, hidden, output_dim))
            self.value_nets.append(ModelWithPrior(MLP(input_dim, hidden, output_dim), 
                                                  MLP(input_dim, hidden, output_dim)))
            self.loss_funcs.append(nn.MSELoss())
            self.optimizers.append(torch.optim.Adam(self.value_nets[-1].parameters(), lr=0.01))

    def _get_reward_from_state(self, s):
        root_z = s[0]
        if root_z > 1.1:
            return 1.0
        else:
            return 1.0 - (1.1 - root_z)

    def learn(self, env):
        for _ in range(self.gradient_steps):
            sampled_idx = np.random.choice(np.min([self.memory_counter, self.memory_size]), size=self.state_samples, replace=False)
            
            sampled_obs = self.obs_mem[sampled_idx,:]
            sampled_targets = self.targets_mem[sampled_idx,:]
            sampled_targets = sampled_targets.transpose()
            
            for i in range(self.num_nets):
                net = self.value_nets[i]
                loss_func = self.loss_funcs[i]
                optimizer = self.optimizers[i]
                
                optimizer.zero_grad()
                
#                 print("Target: {}".format(sampled_targets[i]))
                preds = net(torch.tensor(sampled_obs, dtype=torch.float))
                
                target = torch.tensor([sampled_targets[i]], dtype=torch.float)
                loss = loss_func(preds, target)
            

                loss.backward()
                optimizer.step()
                
                
#             print("=====")
#             print(sampled_obs)
#             print(sampled_targets)
        return
        self.x_init = self.env.sim.get_state()
        
        for _ in range(self.gradient_steps):
            sampled_idx = np.random.choice(np.min([self.memory_counter, self.memory_size]), size=self.state_samples, replace=False)

    #             print(self.state_mem)
    #             print(idx)
    #             sampled_states = self.state_mem[idx]

            sampled_obs = self.obs_mem[sampled_idx,:]

    #             sampled_obs = []

            targets = [None for i in range(self.num_nets)]

            for index in sampled_idx:
                s_state = self.state_mem[index]
                o = self.state_mem[index]
    #             for s_state, o in zip(sampled_states, sampled_obs):


                max_rewards = [float('-inf') for _ in range(self.num_nets)]

                for k in range(self.K):
    #                     print(len(self.x_init), self.x_init)
    #                     print(len(s_state), s_state)
                    self.env.sim.set_state(s_state)
                    discount = 1
                    total_reward = 0
                    for t in range(self.T):
                        perturbed_action_t = self.U[t] + self.noise[k, t]

                        s, reward, _, _ = env.step(np.array([perturbed_action_t]))

                        total_reward += discount * reward
                        discount *= self.gamma

                    for i in range(self.num_nets):
                        net = self.value_nets[i]
                        reward_for_net = torch.tensor(total_reward, dtype=torch.float) + net(torch.tensor(s[:22], dtype=torch.float))
                        if reward_for_net > max_rewards[i]:
                            max_rewards[i] = reward_for_net



                for i in range(self.num_nets):
                    target = max_rewards[i]

                    if targets[i] is None:
                        targets[i] = torch.tensor([[target]], dtype=torch.float)
                    else:
                        targets[i] = torch.cat((targets[i], torch.tensor([[target]], dtype=torch.float)))

        
#         for _ in range(self.gradient_steps):
            for i in range(self.num_nets):
                net = self.value_nets[i]
                loss_func = self.loss_funcs[i]
                optimizer = self.optimizers[i]

                optimizer.zero_grad()

                preds = net(torch.tensor(sampled_obs, dtype=torch.float))

                loss = loss_func(preds, targets[i])

                loss.backward()
                optimizer.step()
                
                
        self.env.sim.set_state(self.x_init)
        
    def get_aggregated_value(self, values):
        weights = softmax(0.01 * values)
        weighted_values = values * weights
        
        return sum(weighted_values)
    
    def get_network_values(self, s):
        values = []
        for net in self.value_nets:
            values.append(net(torch.FloatTensor(s)).tolist())
            
        values = np.array(values)
        return values
    
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
        
        network_values = self.get_network_values(s[:22])
        
        for i in range(self.num_nets):
            reward_for_net = self.reward_total[k] + discount * network_values[i]
            if reward_for_net > self.max_reward_for_net[i]:
                self.max_reward_for_net[i] = reward_for_net
        
        self.reward_total[k] += discount * self.get_aggregated_value(network_values)

    def _ensure_non_zero(self, reward, beta, factor):
        return np.exp(-factor * (beta - reward))


    def get_action_and_targets(self, env):
        self.max_reward_for_net = np.full((self.num_nets), '-inf', dtype=np.float)
        if self.env.unwrapped.spec.id == "Pendulum-v0":
            self.x_init = self.env.env.state
        elif self.env.unwrapped.spec.id == "HumanoidStandup-v2":
            self.x_init = self.env.sim.get_state()
        
        for k in range(self.K):
            self._compute_total_reward(k)
            if self.env.unwrapped.spec.id == "Pendulum-v0":
                self.env.env.state = self.x_init
            elif self.env.unwrapped.spec.id == "HumanoidStandup-v2":
                self.env.sim.set_state(self.x_init)
        print(self.max_reward_for_net)
            
        beta = np.max(self.reward_total)  # maximum reward of all trajectories
        reward_total_non_zero = self._ensure_non_zero(reward=self.reward_total, beta=beta, factor=1/self.lambda_)
        eta = np.sum(reward_total_non_zero)
        omega = 1/eta * reward_total_non_zero
        
        self.U += [np.sum(omega.reshape(len(omega), 1) * self.noise[:, t], axis=0) for t in range(self.T)]
        
        
            
        action = self.U[0]
        
        self.U = np.roll(self.U, -1, axis=0)

        self.U[-1] = self.u_init  #
        self.reward_total[:] = 0
        
        self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, self.env.action_space.shape[0]))
        
        
        
        return action, self.max_reward_for_net
    
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
            if RENDER:
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
                self.x_init = self.env.sim.get_state()
            ###########################
            
            if self.writer is not None:
                self._write_record(timestamp, r, self.U[0], s)
            
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, self.env.action_space.shape[0]))
    
    def write_record(self, timestamp, reward, action, state):
        action_json = json.dumps(action.tolist())
        state_json = json.dumps(state.reshape(len(state), 1).tolist())
        self.writer.writerow([timestamp, reward, action_json, state_json])
        self.log_file.flush()
        
    def store_state(self, obs, state, rewards):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.obs_mem[index] = np.array(obs)
        self.state_mem[index] = state
        self.targets_mem[index] = rewards

        self.memory_counter += 1


# In[7]:


num_epos = 1000
for idx in range(num_epos):
    print("Running for episode {}...".format(idx))
    ENV_NAME = "HumanoidStandup-v2"
    TIMESTEPS = 64  # T
    N_SAMPLES = 128  # K
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    # TIMESTEPS = 15 # T
    # N_SAMPLES = 120  # K

    STATE_SAMPLES = 32

    noise_mu = 0
    noise_sigma = 0.2
    lambda_ = 1.25
    gamma = 0.99

    Z = 16

    env = gym.make(ENV_NAME)

    # from gym.wrappers import Monitor
    # env = Monitor(env, './video', force=True)
    # env._max_episode_steps = 200
    # env.render()
    # env.sim.render(1024, 1024)
    print(env.observation_space)
    print(env.action_space)

    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(TIMESTEPS, env.action_space.shape[0]))  # pendulum joint effort in (-2, +2)
    # print(U)

    log_file = open("polo_record_multiple_{}.tsv".format(idx), "w")

    s = env.reset()

    polo = POLO(env=env, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, 
                    noise_sigma=noise_sigma, u_init=0, memory_size=512, 
                    observation_space=22, action_space=env.action_space.shape[0],
                    state_space=env.observation_space.shape[0], net_hidden_layers=16, 
                    num_nets=6, state_samples=STATE_SAMPLES, gradient_steps=64, 
                    gamma=gamma, log_file=log_file, noise_gaussian=True)


    # polo.store_state(s[:22], env.sim.get_state())

    rewards = []
    for t in range(3000):
        a, targets = polo.get_action_and_targets(env)
        polo.store_state(s[:22], env.sim.get_state(), targets)
        s, r, _, _ = env.step(np.array([a]))
        rewards.append(r)
        polo.write_record(t, s[0], a, s[:22])
        print("episode: {}, timestamp: {}, action taken: {} reward received: {}".format(idx, t, a, s[0]))
        if RENDER:
            env.render()


        if t != 0 and t % Z == 0 and t >= STATE_SAMPLES:
            print("Updating networks...")
            polo.learn(env)


    # mppi_gym.control(iter=30)

    log_file.close()


# In[ ]:




