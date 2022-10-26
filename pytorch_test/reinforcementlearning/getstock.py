import yfinance as yf
import pandas as pd

def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

def get_stock_data_yf(stock_name, past_years, interval) -> pd.DataFrame:
    """
    Get stock data from yahoo finance

    Args:
        stock_name: str, the name of the stock to get data for
        past_years: int, the number of years of data to get
        interval:   str, the interval at which the data is collected. Valid values are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
                    see https://pypi.org/project/yfinance/ for more details
    
    Returns:
        pd.DataFrame, the stock data
    """
    data = yf.download(stock_name, period=f'{past_years}y', interval=interval)
    data = to_numeric_and_downcast_data(data)
    return data