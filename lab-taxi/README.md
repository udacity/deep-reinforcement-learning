# Taxi Problem

### Getting Started

Read the description of the environment in subsection 3.1 of [this paper](https://arxiv.org/pdf/cs/9905014.pdf).  You can verify that the description in the paper matches the OpenAI Gym environment by peeking at the code [here](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py).


### Instructions

The repository contains three files:
- `agent.py`: Develop your reinforcement learning agent here.  This is the only file that you should modify.
- `monitor.py`: The `interact` function tests how well your agent learns from interaction with the environment.
- `main.py`: Run this file in the terminal to check the performance of your agent.

Begin by running the following command in the terminal:
```
python main.py
```

When you run `main.py`, the agent that you specify in `agent.py` interacts with the environment for 20,000 episodes.  The details of the interaction are specified in `monitor.py`, which returns two variables: `avg_rewards` and `best_avg_reward`.
- `avg_rewards` is a deque where `avg_rewards[i]` is the average (undiscounted) return collected by the agent from episodes `i+1` to episode `i+100`, inclusive.  So, for instance, `avg_rewards[0]` is the average return collected by the agent over the first 100 episodes.
- `best_avg_reward` is the largest entry in `avg_rewards`.  This is the final score that you should use when determining how well your agent performed in the task.

Your assignment is to modify the `agents.py` file to improve the agent's performance.
- Use the `__init__()` method to define any needed instance variables.  Currently, we define the number of actions available to the agent (`nA`) and initialize the action values (`Q`) to an empty dictionary of arrays.  Feel free to add more instance variables; for example, you may find it useful to define the value of epsilon if the agent uses an epsilon-greedy policy for selecting actions.
- The `select_action()` method accepts the environment state as input and returns the agent's choice of action.  The default code that we have provided randomly selects an action.
- The `step()` method accepts a (`state`, `action`, `reward`, `next_state`) tuple as input, along with the `done` variable, which is `True` if the episode has ended.  The default code (which you should certainly change!) increments the action value of the previous state-action pair by 1.  You should change this method to use the sampled tuple of experience to update the agent's knowledge of the problem.

Once you have modified the function, you need only run `python main.py` to test your new agent.

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/Taxi-v1/) this task as getting average return of 9.7 over 100 consecutive trials.  
