# My solution for Unity's Banana Navigator environment

### Plot of Rewards
This implementation solved the environment (average reward of +13 over the next 100 consecutive episodes) in 430 episodes. Below is a plot of the rewards during training. 
![Plot of Rewards showing environment solved in 430 episodes](https://github.com/k-staple/deep-reinforcement-learning/blob/write_report/p1_navigation/solved_in_430_episodes.PNG "Plot of Rewards")

### Learning Algorithm
In this project, I implemented the Deep Q-learning algorithm to solve Unity's Banana Navigator environment. At a high-level, this involved an agent with two Deep Q-Networks (also known as a DQN) and a memory buffer interacting with the Unity environment in Navigation.ipynb. Navigation.ipynb contains the code to instantiate the agent and then have the agent store its interactions with the environment as experience in its replay buffer and periodically learn. Learning involves the agent sampling experiences from its replay buffer and using them to train its local Deep Q-Network against a static-during-training target Deep Q-Network. Both of the agent's Deep Q-Networks use the same architecture which was inspired by the original Deep Q-learning paper and can be found in model.py. It involves three convolutional layers and a hidden fully connected layer that each use a ReLU activation function followed by a final fully connected layer. For the convolutional layers, I used a kernael size of two and made the number of output channels four times the number of input channels. For the agent's hyperparameters, I used a learning rate of 5e-4, batch size of 64, replay buffer size of 1e5, gamma (discount factor) of 0.99, and updated the target Deep Q-Netowrk every four timesteps once the replay buffer had a sufficient number of samples for learning.

### Ideas for Future Improvement
To further improve results, I could utilize prioritized replay for the replay buffer and/or implement a double DQN or a dueling DQN.
