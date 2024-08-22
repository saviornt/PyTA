import pandas as pd
import numpy as np
from column_case_solver import solve_case

def EMA(data, span=20, adjust=False):
    """
    Calculate the Exponential Moving Average (EMA) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        span (int): The period over which to calculate the EMA. Default is 20.
        adjust (bool): Whether to adjust the weights of the EMA calculation. Default is False.

    Returns:
        pd.Series: A pandas Series representing the EMA values for the given span.
    """
    data = solve_case(data)
    ema = data['Close'].ewm(span=span, adjust=adjust).mean()
    return ema

def SMMA(data, window=20):
    """
    Calculate the Smoothed Moving Average (SMMA) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the SMMA. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the SMMA values for the given window.
    """
    data = solve_case(data)
    smma = data['Close'].ewm(span=window, adjust=False).mean()
    return smma

def BBANDS(data, window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands (BBANDS).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The period for calculating the moving average. Default is 20.
        num_std_dev (int): The number of standard deviations for the bands. Default is 2.

    Returns:
        pd.DataFrame: DataFrame with 'Middle Band', 'Upper Band', and 'Lower Band'.
    """
    data = solve_case(data)
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return pd.DataFrame({
        'Middle Band': sma,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })

def DEMA(data, span=20):
    """
    Calculate Double Exponential Moving Average (DEMA).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        span (int): The period for calculating the EMA. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the DEMA values.
    """
    data = solve_case(data)
    ema = data['Close'].ewm(span=span, adjust=False).mean()
    dema = 2 * ema - ema.ewm(span=span, adjust=False).mean()
    return dema

def KAMA(data, window=10, fast_ema=2, slow_ema=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods for the efficiency ratio.
        fast_ema (int): The period for the fast EMA constant. Default is 2.
        slow_ema (int): The period for the slow EMA constant. Default is 30.

    Returns:
        pd.Series: A pandas Series representing the KAMA.
    """
    data = solve_case(data)
    change = data['Close'].diff(window).abs()
    volatility = data['Close'].diff().abs().rolling(window=window).sum()
    er = change / volatility
    sc = (er * (2 / (fast_ema + 1) - 2 / (slow_ema + 1)) + 2 / (slow_ema + 1)) ** 2
    kama = data['Close'].ewm(alpha=sc, adjust=False).mean()
    return kama

def TRIMA(data, window=20):
    """
    Calculate Triangular Moving Average (TRIMA).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The period for calculating the TRIMA. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the TRIMA values.
    """
    return data['Close'].rolling(window=window).apply(lambda x: np.mean(np.sort(x)[-window//2:]), raw=True)

def MAMA(data, fast_limit=0.5, slow_limit=0.05):
    """
    Calculate MESA Adaptive Moving Average (MAMA) (simplified version).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        fast_limit (float): The upper limit for the fast moving average. Default is 0.5.
        slow_limit (float): The lower limit for the slow moving average. Default is 0.05.

    Returns:
        pd.Series: A pandas Series representing the MAMA.
    """
    data = solve_case(data)
    return data['Close'].ewm(span=10, adjust=False).mean()  # Simplified version

def MAVP(data, periods):
    """
    Calculate Moving Average with Variable Period (MAVP).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        periods (pd.Series): A pandas Series containing periods to use for each data point.

    Returns:
        pd.Series: A pandas Series representing the MAVP.
    """
    data = solve_case(data)
    mavp = data['Close'].rolling(window=periods).mean()
    return mavp

def MIDPOINT(data, window=14):
    """
    Calculate MidPoint over period.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        window (int): The period for calculating the midpoint. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the midpoint values.
    """
    data = solve_case(data)
    return (data['High'] + data['Low']).rolling(window=window).mean() / 2

def MIDPRICE(data, window=14):
    """
    Calculate Midpoint Price over period.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        window (int): The period for calculating the midpoint price. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the midpoint price values.
    """
    data = solve_case(data)
    return (data['High'] + data['Low']).rolling(window=window).mean()

def SAR(data, af=0.02, max_af=0.2):
    """
    Calculate Parabolic SAR.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        af (float): The acceleration factor. Default is 0.02.
        max_af (float): The maximum acceleration factor. Default is 0.2.

    Returns:
        pd.Series: A pandas Series representing the SAR values.
    """
    data = solve_case(data)
    sar = data['Low'][0]  # Simplified implementation (more advanced logic required)
    trend = 1  # 1 = uptrend, -1 = downtrend
    for i in range(1, len(data)):
        if trend == 1:
            sar = sar + af * (data['High'][i] - sar)
            if data['Low'][i] < sar:
                trend = -1
                sar = data['High'][i]
                af = 0.02
        else:
            sar = sar + af * (sar - data['Low'][i])
            if data['High'][i] > sar:
                trend = 1
                sar = data['Low'][i]
                af = 0.02
        af = min(af + 0.02, max_af)
    return sar

def SAREXT(data, af_start=0.02, af_increment=0.02, af_max=0.2):
    """
    Calculate Parabolic SAR - Extended (SAREXT).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        af_start (float): The starting acceleration factor. Default is 0.02.
        af_increment (float): The increment of the acceleration factor after each period. Default is 0.02.
        af_max (float): The maximum acceleration factor. Default is 0.2.

    Returns:
        pd.Series: A pandas Series representing the SAREXT values.
    """
    data = solve_case(data)
    sar = data['Low'][0]  # Simplified implementation (more advanced logic required)
    trend = 1  # 1 = uptrend, -1 = downtrend
    af = af_start
    sar_series = []

    for i in range(1, len(data)):
        if trend == 1:
            sar = sar + af * (data['High'][i] - sar)
            if data['Low'][i] < sar:
                trend = -1
                sar = data['High'][i]
                af = af_start
        else:
            sar = sar + af * (sar - data['Low'][i])
            if data['High'][i] > sar:
                trend = 1
                sar = data['Low'][i]
                af = af_start
        
        af = min(af + af_increment, af_max)
        sar_series.append(sar)

    return pd.Series(sar_series, index=data.index)

def SMA(data, window=20):
    """
    Calculate the Simple Moving Average (SMA) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the SMA. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the SMA values for the given window.
    """
    data = solve_case(data)
    sma = data['Close'].rolling(window=window).mean()
    return sma

def T3(data, period=5, vfactor=0.7):
    """
    Calculate Triple Exponential Moving Average (T3).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        period (int): The period for calculating the T3. Default is 5.
        vfactor (float): The volume factor. Default is 0.7.

    Returns:
        pd.Series: A pandas Series representing the T3 values.
    """
    data = solve_case(data)
    ema1 = data['Close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()

    c1 = -vfactor**3
    c2 = 3*vfactor**2 + 3*vfactor**3
    c3 = -6*vfactor**2 - 3*vfactor - 3*vfactor**3
    c4 = 1 + 3*vfactor + vfactor**2 + vfactor**3

    t3 = c1*ema3 + c2*ema2 + c3*ema1 + c4*data['Close'].ewm(span=period, adjust=False).mean()
    return t3

def TEMA(data, span=20):
    """
    Calculate Triple Exponential Moving Average (TEMA).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        span (int): The period for calculating the TEMA. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the TEMA values.
    """
    data = solve_case(data)
    ema1 = data['Close'].ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()
    tema = 3 * (ema1 - ema2) + ema3
    return tema

def HT_TRENDLINE(data):
    """
    Calculate the Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE).

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        pd.Series: A pandas Series representing the HT_TRENDLINE values.
    """
    data = solve_case(data)
    close = data['Close']
    n = len(close)
    smooth_price = pd.Series(np.zeros(n))

    # Smoothing constants
    smooth_const = 0.0962
    const_2 = 0.5769

    # Apply smoothing
    for i in range(1, n):
        if i < 5:
            smooth_price[i] = close[i]
        else:
            smooth_price[i] = (smooth_const * close[i] +
                               2 * (1 - smooth_const) * smooth_price[i-1] +
                               (1 - smooth_const) * smooth_price[i-2] - const_2 * smooth_price[i-3])

    # Calculate HT_TRENDLINE
    ht_trendline = pd.Series(np.zeros(n))

    for i in range(1, n):
        if i < 4:
            ht_trendline[i] = smooth_price[i]
        else:
            ht_trendline[i] = 0.5 * (smooth_price[i] + smooth_price[i-4])

    return ht_trendline

def WMA(data, window=20):
    """
    Calculate the Weighted Moving Average (WMA) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The number of periods over which to calculate the WMA. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the WMA values for the given window.
    """
    data = solve_case(data)
    weights = np.arange(1, window + 1)
    wma = data['Close'].rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

def MA(data, window=20, method='sma'):
    """
    Calculate Moving Average (MA) using the specified method.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        window (int): The period for calculating the moving average. Default is 20.
        method (str): The type of moving average ('sma', 'ema', 'wma', 'trima', 'tema', 'dema', 'kama', 't3', 'mavp', 'mama', 'smma'). Default is 'sma'.

    Returns:
        pd.Series: A pandas Series representing the MA values.
    """
    match method.lower():
        case 'sma':
            return SMA(data, window=window)
        case 'ema':
            return EMA(data, span=window)
        case 'wma':
            return WMA(data, window=window)
        case 'trima':
            return TRIMA(data, window=window)
        case 'tema':
            return TEMA(data, span=window)
        case 'dema':
            return DEMA(data, span=window)
        case 'kama':
            return KAMA(data, window=window)
        case 't3':
            return T3(data, period=window)
        case 'mavp':
            return MAVP(data, periods=window)  # Assuming window is periods
        case 'mama':
            return MAMA(data)
        case 'smma':
            return SMMA(data, window=window)
        case 'ht_trendline':
            return HT_TRENDLINE(data)
        case _:
            raise ValueError(f"Unknown method: {method}")
