"""
This is a custom class use for animating stock trading data and agent action from an open ai gym environment.
There will be two subplots, one illustrates the net worth of the agent;
And the other illustrates the stock price along with the agent action.

It will utilize the new mplfinance library to plot the candlestick chart of the stock price.
Simply line chart will be used to plot the net worth of the agent.

matplotlib.animation will be used to animate the chart.

To use this class, create an instance of the class, pass the dataframe with the stock price data,
When animating, call the animate function with current_step, list of net_worth, list of trades action so far and windows_size
"""

import mplfinance as mpf



class StockTradingGraph:
    def __init__(self, df, dfvolume, action_history, net_worth_history, windows_size=45):
        # Save the dataframe with the stock price data (get only open, high, low, close from df)
        self.df = df[['Open', 'High', 'Low', 'Close']]
        # get dfvolume as a column of df
        self.df['Volume'] = dfvolume

        self.net_worth = net_worth_history

        self.action_history = action_history

        self.windows_size = windows_size


    def plot(self, current_step):
        # Get the data for the current window without the networth column
        start = max(current_step - self.windows_size, 0)
        end = current_step + 1
        data = self.df.iloc[start:end]

        # create a new column for sell marker position (slightly above the high price when the action history indicates sell)
        sell = data['High'].where(-1 <= self.action_history[0] <= -2/3)*1.05
        # create a new column for buy marker position (slightly below the low price when the action history indicates buy)
        buy = data['Low'].where(1 >= self.action_history[0] >= 2/3)*0.95

        # add networth line chart to subplot
        net_worth_ap = mpf.make_addplot(self.net_worth, type='line', ylabel='Net Worth ($)', panel = 2)
        # add buy marker to subplot
        buy_ap = mpf.make_addplot(buy, type='scatter', marker='^', markersize=100, color='green', panel = 0)
        # add sell marker to subplot
        sell_ap = mpf.make_addplot(sell, type='scatter', marker='v', markersize=100, color='red', panel = 0)

        # create the fig
        fig, axlist = mpf.plot(data, type='candle', addplot=[net_worth_ap, buy_ap, sell_ap], volume=True, 
                               returnfig=True, volume_panel=1, style='yahoo', ylabel='Price ($)', ylabel_lower='Shares\nTraded')
        
        # return the fig
        return fig

