# library of function for Support Vector Classification on stock data

# import library for Support Vector Classification
from sklearn.svm import SVC

import pandas as pd
import numpy as np

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

    return model, df_train['High-Low'], df_train['Open-Close']