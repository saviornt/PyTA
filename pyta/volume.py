import pandas as pd
import numpy as np
from column_case_solver import solve_case

def AD(high, low, close, volume):
    """
    Calculate the Chaikin A/D Line (AD).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        volume (pd.Series): The volume.

    Returns:
        pd.Series: A pandas Series representing the AD values.
    """
    ad = ((2 * close - high - low) / (high - low)) * volume
    ad_line = ad.cumsum()
    return ad_line

def ADL(data):
    """
    Calculate the Accumulation/Distribution Line (ADL) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with 'Close', 'High', 'Low', and 'Volume' columns.

    Returns:
        pd.Series: A pandas Series representing the ADL values.
    """
    data = solve_case(data)
    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    adl = (clv * data['Volume']).cumsum()
    return pd.Series(adl, name='ADL')

def ADOSC(data, short_period=3, long_period=10):
    """
    Calculate the Chaikin A/D Oscillator (ADOSC).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        volume (pd.Series): The volume.
        short_period (int): The short period for the moving average. Default is 3.
        long_period (int): The long period for the moving average. Default is 10.

    Returns:
        pd.Series: A pandas Series representing the ADOSC values.
    """
    data = solve_case(data)
    ad_line = AD(data['High'], data['Low'], data['Close'], data['Volume'])
    adosc = ad_line.rolling(window=short_period).mean() - ad_line.rolling(window=long_period).mean()
    return adosc

def OBV(data):
    """
    Calculate the On-Balance Volume (OBV) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns.

    Returns:
        pd.Series: A pandas Series representing the OBV values.
    """
    data = solve_case(data)
    obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                   np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0)).cumsum()
    return pd.Series(obv, name='OBV')

def VWAP(data):
    """
    Calculate the Volume Weighted Average Price (VWAP) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with 'Close', 'Volume', 'High', and 'Low' columns.

    Returns:
        pd.Series: A pandas Series representing the VWAP values.
    """
    data = solve_case(data)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return pd.Series(vwap, name='VWAP')

