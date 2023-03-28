"""
this trading environment class is able to take in a time series of stock data with abritrary number of features and create state space
the step function will take in an action (which is the number of shares to buy or sell) and output the next state, reward and done
the reset function will reset the environment to the initial state

see https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e for more details
"""

MAX_ACCOUNT_BALANCE = 2147483647

# import the necessary packages
import gym
from gym.envs.registration import register
from gym import spaces, error, utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# helper function to get stock data
from getstock import *

# define the trading environment class
class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, init_balance, max_step, random = True):
        super(StockTradingEnv, self).__init__()

        # data
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.init_balance = init_balance
        self.max_step = max_step
        self.random = random

        # action space (buy x%, sell x%, hold)
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # observation space (prices and technical indicators)
        # shape should be (n_features + 6) where 6 is the number of additional dynamic features of the environment
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.df.columns)+6,), dtype=np.float16)

    # reset the state of the environment to an initial state
    def reset(self):
        self.balance = self.init_balance
        self.net_worth = self.init_balance
        self.max_net_worth = self.init_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
        if self.random:
            # set the current step to a random point within the data frame
            self.current_step = np.random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        else:
            self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        # get the features from the data frame for current time step
        frame = self.df.iloc[self.current_step].values

        # append additional data
        obs = np.append(frame, [
            self.balance,
            self.max_net_worth,
            self.shares_held,
            self.cost_basis,
            self.total_shares_sold,
            self.total_sales_value,
        ], axis=0)

        return obs

    def step(self,action):
        
        # Set the execute_price to the closing price of the time step
        execute_price = self.df.iloc[self.current_step]["Close"]
        # Execute one time step within the environment
        self._take_action(action,execute_price)
        self.current_step += 1
        # calculate reward based on the balance with a delay modifier. which bias towards having a higher balance towards the end of the episode
        delay_modifier = (self.current_step / self.max_step)
        reward = self.balance * delay_modifier
        # if net_worth is below 0, or current_step is greater than max_step, then done = True
        done = self.net_worth <= 0 or self.current_step >= self.max_step

        obs = self._next_observation()

        return obs, reward, done, {}
    
    def _take_action(self,action, execute_price):
        # Set the current price to a random price within the time step

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # buy amount % of balance in shares
            total_possible = self.balance / execute_price
            # shares bought rounded to integer
            shares_bought = int(total_possible * amount)

            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * execute_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # sell amount % of shares held (rounded to interger)
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * execute_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * execute_price

        self.net_worth = self.balance + self.shares_held * execute_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - self.init_balance

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        