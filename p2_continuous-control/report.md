# Project 2: Continuous Control

### Introduction

The task at hand is to solve the Unity ML Agents [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

The environment simulates a double-jointed arm, the goal being to train said arm to move to a target location. The agent receives a positive reward of 0.1 for every time step it manages to remain in the goal location. This reward setup guarantees therefore that the arm will eventually learn to go to and stay in that location for as many timesteps as possible. The task is considered solved if the agent manages to consistently amass 30 points of reward in any episode. For practical purposes, we will take that to mean an average reward mean of +30 over 100 consecutive episodes. 

The state space is discrete and consists of 33 variables describing the position, rotation, velocity, and agular velocities of the arm. The agent can modulate an action vector of 4 continuous variables that take on values in the interval \[-1, 1\]. These 4 values correspond to the torque that the agent can apply to its joints to act on the environment and try and maximize expected future reward. 

The continous nature of the action space makes a policy-based approach the preferred one. The project could be solved in one of two variants: either a multi-agent approach, or the usual 1-agent approach. This solution adopted the latter approach. 

### Solution 


Seeing as the action space necessitates the use of a policy-based method, the DDPG (Deep deterministic policy gradient) approach is chosen due to its simplicity, and stability. DDPG is an actor-critic method that borrows heavily from its successful value-based cousin, DQN. Both the Actor and Critic networks have two hidden layers with 400 and 300 nodes each and activated with rectified linear units. The weights between the local networks are constantly leaked into those of the target network using a soft update with an interpolation coefficient tau of 1e-3.
Further stabilitiy is achieved by calling the learning algorithms 10 times every 20 steps. The learning rate of the actor is half that of the critic, respectively 5e-5 and 1e-4. Adam is used to optimize the gradient descent steps. To add an element of exploration, noise is sampled from an Ornstein-Uhlenbeck process with parameters theta=0.15 and sigma=0.2.

I had to run the whole process for a few thousand times on a GPU, reducing the magnitude of the noise by 20% to emphasize exploitation the more experience was gathered and studied. The learning was extremely slow, but also extremely stable. 
