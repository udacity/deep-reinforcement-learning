# Project 3: Collaboration and Competition



### Introduction

The task at hand is to solve the Unity ML Agents [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

The environment simulates two tennis players who must work together to send the ball back and forth over the net without sending said ball out of bounds or dropping it. The agents share one Unity brain. Each agent observes a state of 24 continous variables and can take two actions, both continous, corresponding to horizontal movement and jumping. Each agent receives a reward of +0.1 when they successfully hit the ball over the net, and one of -0.1 when they the ball hits their ground or when they drive the ball out of bounds. 


The continous nature of the action space makes a policy-based approach the preferred one. The fact that two agents need to cooperate to solve the task makes it necessary to use a multi-agent approach.  

### Solution 


Seeing as the action space necessitates the use of a policy-based method, the MADDPG (Multi Agent Deep deterministic policy gradient) approach is chosen due to its simplicity, and stability. MADDPG is the multi-agent variant of DDPG which is an actor-critic method that borrows heavily from its successful value-based cousin, DQN. This method sees each agent's critic having access to all the information that is available to the each agent during training, but not during inference. 
Both the Actor and Critic networks for each agent have two hidden layers with 256 and 256 nodes each and activated with rectified linear units. The weights between the local networks are constantly leaked into those of the target network using a soft update with an interpolation coefficient tau of 1e-3. The first hidden layer is always batch-normalized. Adam optimizers are used for all networks, and the learning rate of the critic is 10 times that of the actor.
To add an element of exploration, noise is sampled from an Ornstein-Uhlenbeck process with parameters theta=0.15 and sigma=0.2 and its norm halved. Future rewards are not discounted which guarantees a forward looking agent. 

I had to run the whole process for a a little under 3000 episodes on a moderately powerful GPU. The learning was overall stable, though at times jittery. 


### Improvements

Two major improvements are those suggested by the original MADDPG paper, first namely to infer the policies of other agents instead of forcing the assumption of the critics just knowing everything about the envinronment and other agents. Second, and to borrow from other ML successes, would be to use ensembles which combats the non-stationarity which is caused by other agents, and therefore the envinronment changing the policies. This causes the learning to overfit to the current adversary/partner and poorly generalize. A third improvement could be to implement a human-in the loop approach and use imitation learning. That is, to have one of the two agents be a human player for a while, and have the agent bootstrap off of what a human would do. 