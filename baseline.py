import numpy as np

class NaiveHeuristicAgent:
    def __init__(self, env):
        self.env = env
        self.price_history = []

    def update_price_history(self, current_price):
        self.price_history.append(current_price)
        if len(self.price_history) > 24*30:  # Keeping last 30 days of prices
            self.price_history.pop(0)

    def choose_action(self):
        current_price = self.env.state[1]
        self.update_price_history(current_price)
        
        lower_quantile = np.quantile(self.price_history, 0.25)
        upper_quantile = np.quantile(self.price_history, 0.75)

        if current_price < lower_quantile and abs(self.env.battery_level - self.env.battery_capacity) > 0.0001:
            return min(1,(self.env.battery_capacity - self.env.battery_level) / (self.env.max_power * 0.9))
        elif current_price > upper_quantile and abs(self.env.battery_level) > 0.0001:
            return max(-1,-self.env.battery_level / (self.env.max_power * 0.9))
        else:
            return 0  # No action