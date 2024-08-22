import pandas as pd
import numpy as np
from column_case_solver import solve_case

def ATR(data, period=14):
    """
    Calculate the Average True Range (ATR) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
        period (int): The number of periods to calculate ATR. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ATR values.
    """
    data = solve_case(data)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return pd.Series(atr, name='ATR')

def NATR(data, window=14):
    """
    Calculate Normalized Average True Range (NATR).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window (int): The period over which to calculate the average true range. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the normalized average true range values.
    """
    data = solve_case(data)
    tr = TRANGE(data)
    atr = tr.rolling(window=window).mean()
    natr = atr / data['Close'].rolling(window=window).mean() * 100
    return natr

def TRANGE(data):
    """
    Calculate True Range (TRANGE).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series representing the true range values.
    """
    data = solve_case(data)
    high_low = data['High'] - data['Low']
    high_prev_close = (data['High'] - data['Close'].shift()).abs()
    low_prev_close = (data['Low'] - data['Close'].shift()).abs()
    
    tr = pd.DataFrame({
        'High_Low': high_low,
        'High_Prev_Close': high_prev_close,
        'Low_Prev_Close': low_prev_close
    }).max(axis=1)
    
    return tr