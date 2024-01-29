import numpy as np

class TabularQLearningAgent:
    def __init__(self, alpha, gamma, epsilon, battery_segments, num_actions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.battery_segments = battery_segments
        self.num_actions = num_actions
        self.q_table = np.zeros((len(battery_segments), num_actions))

    def discretize_state(self, state):
        battery_level = state[0]
        battery_bin = np.digitize(battery_level, self.battery_segments, right=False) - 1
        return battery_bin

    def choose_action(self, state):
        discrete_state = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        best_next_action = np.argmax(self.q_table[discrete_next_state])
        self.q_table[discrete_state, action] += self.alpha * (reward + self.gamma * self.q_table[discrete_next_state, best_next_action] - self.q_table[discrete_state, action])
        print(self.q_table)