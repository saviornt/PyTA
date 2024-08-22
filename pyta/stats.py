import pandas as pd
import numpy as np
from scipy.stats import linregress
from column_case_solver import solve_case

def BETA(data, market_data, window=20):
    """
    Calculate Beta.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column for the asset.
        market_data (pd.DataFrame): DataFrame containing at least a 'Close' column for the market index.
        window (int): The number of periods over which to calculate Beta. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the Beta values.
    """
    data = solve_case(data)
    returns = data['Close'].pct_change()
    market_returns = market_data['Close'].pct_change()
    
    covariance = returns.rolling(window=window).cov(market_returns)
    variance = market_returns.rolling(window=window).var()
    
    beta = covariance / variance
    return beta

def CORREL(data, market_data, window=20):
    """
    Calculate Pearson's Correlation Coefficient (r).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column for the asset.
        market_data (pd.DataFrame): DataFrame containing at least a 'Close' column for the market index.
        window (int): The number of periods over which to calculate the correlation. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the correlation coefficient.
    """
    data = solve_case(data)
    returns = data['Close'].pct_change()
    market_returns = market_data['Close'].pct_change()
    
    correlation = returns.rolling(window=window).corr(market_returns)
    return correlation

def LINEARREG(data, window=20):
    """
    Calculate Linear Regression.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the linear regression. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the linear regression values.
    """
    data = solve_case(data)
    rolling_window = data['Close'].rolling(window=window)
    linear_reg = rolling_window.apply(lambda x: linregress(np.arange(len(x)), x)[0] * len(x) + linregress(np.arange(len(x)), x)[1], raw=True)
    return linear_reg

def LINEARREG_ANGLE(data, window=20):
    """
    Calculate Linear Regression Angle.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the linear regression angle. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the angle of the linear regression line in degrees.
    """
    data = solve_case(data)
    rolling_window = data['Close'].rolling(window=window)
    angle = rolling_window.apply(lambda x: np.degrees(np.arctan(linregress(np.arange(len(x)), x)[0])), raw=True)
    return angle

def LINEARREG_INTERCEPT(data, window=20):
    """
    Calculate Linear Regression Intercept.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the linear regression intercept. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the intercept of the linear regression line.
    """
    data = solve_case(data)
    rolling_window = data['Close'].rolling(window=window)
    intercept = rolling_window.apply(lambda x: linregress(np.arange(len(x)), x)[1], raw=True)
    return intercept

def LINEARREG_SLOPE(data, window=20):
    """
    Calculate Linear Regression Slope.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the linear regression slope. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the slope of the linear regression line.
    """
    data = solve_case(data)
    rolling_window = data['Close'].rolling(window=window)
    slope = rolling_window.apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=True)
    return slope

def STDDEV(data, window=20):
    """
    Calculate Standard Deviation.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the standard deviation. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the standard deviation values.
    """
    data = solve_case(data)
    stddev = data['Close'].rolling(window=window).std()
    return stddev

def TSF(data, period=14):
    """
    Calculate Time Series Forecast (TSF).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        period (int): The number of periods to forecast. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the time series forecast values.
    """
    data = solve_case(data)
    rolling_window = data['Close'].rolling(window=period)
    tsf = rolling_window.apply(lambda x: linregress(np.arange(len(x)), x)[0] * (len(x) + period - 1) + linregress(np.arange(len(x)), x)[1], raw=True)
    return tsf

def VAR(data, window=20):
    """
    Calculate Variance.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the variance. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the variance values.
    """
    data = solve_case(data)
    var = data['Close'].rolling(window=window).var()
    return var
