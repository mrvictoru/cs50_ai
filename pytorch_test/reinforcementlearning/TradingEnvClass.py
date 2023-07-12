"""
this trading environment class is able to take in a time series of stock data with abritrary number of features and create state space
the step function will take in an action (which is the number of shares to buy or sell) and output the next state, reward and done
the reset function will reset the environment to the initial state

see https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e for more details
"""

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'
VOLUME_CHART_HEIGHT = 0.33


LOOKBACK_WINDOW_SIZE = 60


# import the necessary packages
import gym
from gym.envs.registration import register
from gym import spaces, error, utils

import numpy as np
import pandas as pd
import matplotlib.animation as animation
import sys

# helper function to get stock data
from getstock import *

# Candlestick graph class
from StockTradingGraph import StockTradingGraph

# define the trading environment class
# This class defines a gym environment for simulating stock trading. The environment takes a pandas DataFrame of stock prices as input, 
# along with an initial account balance, a maximum number of steps, and a flag indicating whether to start at a random point in the data frame. 
# The environment provides an action space for buying, selling, or holding shares, and an observation space consisting of the current stock prices and additional features such as the current account balance and net worth. 
# The environment also provides a reward function based on the account balance and a delay modifier, which biases the reward towards having a higher balance towards the end of the episode. 
# The environment can be reset to an initial state, and can step forward in time by executing an action. 
# The environment provides a render function for displaying the current state of the environment, and a metadata attribute for specifying the available render modes.

# Example usage:
# import gym
# import pandas as pd
# from TradingEnvClass import StockTradingEnv

# load stock price data
# df = pd.read_csv('stock_prices.csv')

# create trading environment
# env = StockTradingEnv(df, init_balance=10000, max_step=1000, random=True)

# reset environment to initial state
# obs = env.reset()

# loop over steps
# for i in range(1000):
#     # choose random action
#     action = env.action_space.sample()
#     # step forward in time
#     obs, reward, done, info = env.step(action)
#     # render environment
#     env.render()
#     # check if episode is done
#     if done:
#         break

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'None']}
    visualization = None

    def __init__(self, df, init_balance, max_step, random = True):
        super(StockTradingEnv, self).__init__()

        # data
        # get all the features from df except for the column 'Volume'
        self.df = df.drop(columns=['Volume'])
        self.dfvolume = df['Volume']
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.init_balance = init_balance
        self.max_step = max_step
        self.random = random

        self.net_worths = []

        # normalize the data
        self.price_mean = self.df['Close'].mean()
        self.price_std = self.df['Close'].std()
        self.df_standard = (df - df.mean()) / df.std()

        # trade action history
        self.action_history = []

        # action space (buy x%, sell x%, holdclass StockTradingEnv(gym.Env):

        self.action_space = spaces.Box(low=np.array([-1, 0.01]), high=np.array([1, 0.99]), dtype=np.float16)

        # observation space (prices and technical indicators)
        # shape should be (n_features + 6) where 6 is the number of additional dynamic features of the environment
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(df.columns)+6,), dtype=np.float16)

    # reset the state of the environment to an initial state
    def reset(self):
        self.balance = self.init_balance
        self.net_worth = self.init_balance
        self.max_net_worth = self.init_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.action_history = []
        self.net_worths = []
        
        if self.random:
            # set the current step to a random point within the data frame
            self.current_step = np.random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        else:
            self.current_step = 0
        return self._next_observation_norm()

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
    
    def _next_observation_norm(self):
        # get the features from the data frame for current time step
        frame = self.df_standard.iloc[self.current_step].values

        # normalize the additional data to avoid gradient issues.
        # append additional data
        obs = np.append(frame, [
            self.balance/MAX_ACCOUNT_BALANCE,
            self.max_net_worth/MAX_ACCOUNT_BALANCE,
            self.shares_held/MAX_NUM_SHARES,
            (self.cost_basis - self.price_mean)/self.price_std,
            self.total_shares_sold/MAX_NUM_SHARES,
            self.total_sales_value/(MAX_NUM_SHARES *MAX_SHARE_PRICE),
        ], axis=0)

        return obs

    def step(self,action):
        
        # Set the execute_price to the closing price of the time step
        execute_price = self.df.iloc[self.current_step]["Close"]
        # Execute one time step within the environment
        self._take_action(action,execute_price)
        self.current_step += 1
        self.net_worths.append(self.net_worth)
        self.action_history.append(action)
        # calculate reward based on the net worth/balance with a delay modifier. which bias towards having a higher balance towards the end of the episode
        delay_modifier = (self.current_step / self.max_step)
        reward = self.balance * delay_modifier
        # reward = self.net_worth * delay_modifier 
        # if net_worth is below 0, or current_step is greater than max_step, then environment terminates
        done = self.net_worth <= 0 or self.current_step >= self.max_step

        obs = self._next_observation_norm()

        return obs, reward, done, {}
    
    def _take_action(self,action, execute_price):
        # Set the current price to a random price within the time step

        action_type = action[0]
        amount = action[1]

        # check if action_type between 2/3 and 1 then it is to buy
        if 2/3 <= action_type <= 1:
        
            # buy amount % of balance in shares
            total_possible = self.balance / execute_price
            # shares bought rounded to integer
            shares_bought = int(total_possible * amount)
            # if shares bought is 0 then make it one
            if shares_bought < 1:
                shares_bought = 1  

            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * execute_price

            self.balance -= additional_cost
            # calculate the new cost basis, check if it is divide by zero, if it is then set it to the execute price
            if self.shares_held + shares_bought == 0:
                self.cost_basis = execute_price
            else:
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            
            self.shares_held += shares_bought


        elif -1 <= action_type <= -2/3:
            # sell amount % of shares held (rounded to interger)
            shares_sold = int(self.shares_held * amount)
            # if shares sold is 0 then make it one unless we have no shares
            if shares_sold < 1:
                shares_sold = 1
            self.balance += shares_sold * execute_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * execute_price
            

        self.net_worth = self.balance + self.shares_held * execute_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
          
    # see https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4        
    def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - self.init_balance

        file = open(filename, 'a+')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        file.write(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n')
        file.write(f'Action: {self.action_history[-1]}\n')
        file.close()

    def render(self, mode='None', **kwargs):
        # Render the environment to the screen
        profit = self.net_worth - self.init_balance
        if mode == 'print':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
            print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
            print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            # print out the current stock price
            print(self.df.iloc[self.current_step])
        elif mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))
        elif mode == 'plot':
            if self.visualization == None:
                self.visualization = StockTradingGraph(self.df, self.dfvolume, self.action_history, self.net_worths, windows_size=LOOKBACK_WINDOW_SIZE)
            if self.current_step > LOOKBACK_WINDOW_SIZE:
                return self.visualization.plot(self.current_step)

        else:
            # return the observation
            return self._next_observation()
        return None


    