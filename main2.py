from TestEnv import Electric_Car
from baseline import NaiveHeuristicAgent
from agent import TabularQLearningAgent
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Make the excel file as a command line argument, so that you can do: " python3 main.py --excel_file validate.xlsx "
parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
args = parser.parse_args()

env = Electric_Car(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []
battery_level = []

training_mode = True
observation = env.observation()
# agent = NaiveHeuristicAgent(env)
agent = TabularQLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, battery_segments=np.linspace(0, 50, 6), num_actions=5)
action_mapping = {0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1}  # Mapping Q-learning actions to environment actions

for i in range(730*24-1): # Loop through 2 years -> 730 days * 24 hours - 1
    state = env.observation()
    q_action = agent.choose_action(state)  # Q-learning action
    env_action = action_mapping[q_action]  # Convert to corresponding environment action
    next_observation, reward, terminated, truncated, info = env.step(env_action)
    
    print(state[0], q_action, env_action, reward, next_observation[0])
    # Training: Update Q-table
    if training_mode:
        agent.update_q_table(state, q_action, reward, next_observation)
        
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))
    done = terminated or truncated
    observation = next_observation

    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.ylabel('Cumulative reward (€)')
        plt.show()