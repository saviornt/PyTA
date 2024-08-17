import pandas as pd
import numpy as np

def AVGPRICE(data):
    """
    Calculate Average Price (AVGPRICE).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series representing the average price.
    """
    return (data['High'] + data['Low'] + data['Close']) / 3

def MEDPRICE(data):
    """
    Calculate Median Price (MEDPRICE).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

    Returns:
        pd.Series: A pandas Series representing the median price.
    """
    return (data['High'] + data['Low']) / 2

def PP(data):
    """Calculates Pivot Points and their associated support and resistance levels for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns

    Returns:
        pd.DataFrame: A DataFrame with Pivot Point (PP), and support (S1, S2, S3) and resistance (R1, R2, R3) levels.
    """
    pivot = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
    resistance1 = (2 * pivot) - data['Low'].shift(1)
    support1 = (2 * pivot) - data['High'].shift(1)
    resistance2 = pivot + (data['High'].shift(1) - data['Low'].shift(1))
    support2 = pivot - (data['High'].shift(1) - data['Low'].shift(1))
    
    data['Pivot'] = pivot
    data['Resistance1'] = resistance1
    data['Support1'] = support1
    data['Resistance2'] = resistance2
    data['Support2'] = support2
    
    return data[['Pivot', 'Resistance1', 'Support1', 'Resistance2', 'Support2']]

def TYPPRICE(data):
    """
    Calculate Typical Price (TYPPRICE).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series representing the typical price.
    """
    return (data['High'] + data['Low'] + data['Close']) / 3

def WCLPRICE(data):
    """
    Calculate Weighted Close Price (WCLPRICE).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series representing the weighted close price.
    """
    return (data['High'] + data['Low'] * 2 + data['Close']) / 4
