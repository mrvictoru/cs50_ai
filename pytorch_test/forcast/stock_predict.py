import torch
import torch.nn as nn
import pandas as pd
import pandas_datareader as web
import numpy as np
# import library for Support Vector Classification
from sklearn.svm import SVC

# define custom dataset
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target, features, seq_len = 60):
        self.dataframe = dataframe
        self.target = target
        self.seq_len = seq_len
        self.x = torch.tensor(dataframe[features].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target].values, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if idx >= self.seq_len - 1:
            idx_start = idx - self.seq_len + 1
            x = self.x[idx_start:idx+1,:]
        else:
            padding = self.x[0].repeat(self.seq_len - idx - 1, 1)
            x = self.x[0:(idx+1),:]
            x = torch.cat((padding, x), dim=0)

        return x, self.y[idx]

# define model
class ShallowRegressionLSTM(nn.Module):
    def __init__(self,n_features, n_hidden, n_layers=1, device='cpu'):
        super(ShallowRegressionLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size = n_features, 
            hidden_size=n_hidden, 
            num_layers=n_layers, 
            batch_first=True)
        self.linear = nn.Linear(in_features = self.n_hidden, out_features=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_().to(self.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_().to(self.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten() #first dim of hn is the layer dimension
        return out

# define training model function
def train(model, loader, loss, optimizer, device):
    model.train()
    train_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        batch_loss = loss(y_pred, y)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    avg_loss = train_loss / len(loader)
    print(f"Train loss: {avg_loss:.4f}")

# define testing model function
def test(model, loader, loss, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            batch_loss = loss(y_pred, y)
            test_loss += batch_loss.item()

    avg_loss = test_loss / len(loader)
    print(f"Test loss: {avg_loss:.4f}")

# get predictions
def predict(loader, model, device):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_pred = model(x)
            output = torch.cat((output, y_pred), dim=0)
    
    return output.cpu().numpy()

# get predictions for binary action
def predict_binary(loader, model, device):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)
            # convert to binary
            y_pred = torch.round(y_pred)
            output = torch.cat((output, y_pred), dim=0)
    
    return output.cpu().numpy()

# define read web stock data function
def get_stock_data(stock_name = 'TSLA', source = 'yahoo', past_year = 2):
    # Get the past stock price
    # today's date
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    # get start date
    start = pd.to_datetime('today') - pd.DateOffset(years=past_year)
    return web.DataReader(stock_name, data_source=source, start=start, end=today)

# define function to split train and test data
def split_train_test(training_target = "Close", df = pd.DataFrame(), train_ratio = 0.8):
    df_split = df.copy()
    feature = list(df.columns.difference([training_target]))

    forecast_lead = 1
    target = f"{training_target}_t+{forecast_lead}"

    df_split[target] = df_split[training_target].shift(-forecast_lead)
    df_split = df_split.iloc[:-forecast_lead]

    # split data
    train_size = int(len(df_split) * train_ratio)
    test_size = len(df_split) - train_size
    train, test = df_split.iloc[0:train_size].copy(), df_split.iloc[train_size:len(df)].copy()

    # standardize data
    target_mean = train[target].mean()
    target_std = train[target].std()

    for c in train.columns:
        mean = train[c].mean()
        std = train[c].std()

        train[c] = (train[c] - mean) / std
        test[c] = (test[c] - mean) / std
    
    return train, test, feature, target, train_size, test_size, target_mean, target_std

# define function to split train and test data with binary action as target
def split_train_test_binary(df = pd.DataFrame(), train_ratio = 0.8):
    df_split = df.copy()
    # create binary action target
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df_split['perfect'] = y

    # split data
    train_size = int(len(df_split) * train_ratio)
    test_size = len(df_split) - train_size
    train, test = df_split.iloc[0:train_size].copy(), df_split.iloc[train_size:len(df)].copy()

    # standardize data

    for c in train.columns:
        if c != 'perfect':
            mean = train[c].mean()
            std = train[c].std()

            train[c] = (train[c] - mean) / std
            test[c] = (test[c] - mean) / std
    
    return train, test, train_size, test_size, y

def svc_train(df):
    # create copy of dataframe
    df_train = df.copy()
    # Create the independent variables
    df_train['High-Low'] = df_train['High'] - df_train['Low']
    df_train['Open-Close'] = df_train['Open'] - df_train['Close']
    # Store the independent variables in a new variable called 'X'
    X = df_train[['High-Low', 'Open-Close', 'Close']]
    # Store target variable in a new variable called 'y': if tomorrows close price is greater than todays close price, then y = 1, else y = 0
    # 1 indicate to buy by today closing and sell by tomorrow closing and 0 indicates no action
    y = np.where(df_train['Close'].shift(-1) > df_train['Close'], 1, 0)

    # Get the percentage to split the data into training (90%) and testing sets (10%)
    split_percentage = 0.9
    row = int(df_train.shape[0] * split_percentage)

    # Cretate the training data set
    X_train = X[:row]
    y_train = y[:row]

    # Create model
    model = SVC()

    # Train the model
    model.fit(X_train[['Open-Close','High-Low']], y_train)

    return model

