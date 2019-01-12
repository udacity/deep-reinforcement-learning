from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

plt.figure(figsize=(20, 10))
plt.plot(avg_rewards)
plt.show()
