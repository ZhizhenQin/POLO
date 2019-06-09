import gym
import numpy as np
import csv
import json 

ENV_NAME = "HumanoidStandup-v2"
# ENV_NAME = "Ant-v2"

# ENV_NAME = "Acrobot-v1"
# ENV_NAME = "Pendulum-v0"
env = gym.make(ENV_NAME)
env.reset()
from gym.wrappers import Monitor
# env = Monitor(env, './video', force=True)
env.reset()


env.render()
for _ in range(200):
    # init_state = env.sim.get_state()
    # for i in range(10):
    #     U = np.random.uniform(low=-1.0, high=1.0, size=(env.action_space.shape[0]))
    #     env.step(U)
    # env.sim.set_state(init_state)
    
    # U = np.random.uniform(low=-1.0, high=1.0, size=(env.action_space.shape[0]))
    env.step(0)
    import time
    time.sleep(0.05)
    env.render()