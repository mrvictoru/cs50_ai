# utility function for creating a custom trading environment

# customise the trading environment signal features
def process_data(df,window_size, frame_bound, price_feature, signal_features):
    start = frame_bound[0] - window_size
    end = frame_bound[1]
    prices = df.loc[:, price_feature].to_numpy()[start:end]
    signals = df.loc[:, signal_features].to_numpy()[start:end]
    return prices, signals

# create a gym environment from loading sliding dataset and sample state, action, reward
import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

# define the custom environment with signal features
class CustomStocksEnv(StocksEnv):
    def __init__(self, prices, signal_features, **kwargs):
        self._prices = prices
        self._signal_features = signal_features
        super().__init__(**kwargs)

    def _process_data(self):
        return self._prices, self._signal_features

# create a custom dataset class that can provide sliding window data
import torch
from torch.utils.data import Dataset

class SlidingDataset(Dataset):
    def __init__(self, data, window_size, stride=1, transform=None):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.transform = transform

    def __len__(self):
        return (len(self.data) - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = idx * self.stride
        end = start + self.window_size
        sample = self.data[start:end]

        if self.transform:
            sample = self.transform(sample)

        return sample