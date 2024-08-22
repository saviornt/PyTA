import pandas as pd
import numpy as np
from column_case_solver import solve_case

def ADX(data, period=14):
    """
    Calculate the Average Directional Index (ADX) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The number of periods to calculate ADX. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ADX values.
    """
    data = solve_case(data)

    high = data['High']
    low = data['Low']
    close = data['Close']

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = abs(100 * (minus_dm.rolling(window=period).mean() / atr))

    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()

    return pd.Series(adx, name='ADX')

def ADXR(data, period=14):
    """
    Calculate the Average Directional Movement Index Rating (ADXR).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the ADX. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ADXR values.
    """
    data = solve_case(data)
    adx = ADX(data['High'], data['Low'], data['Close'], period)
    adxr = (adx + adx.shift(period)) / 2
    return adxr

def APO(data, fastperiod=12, slowperiod=26, matype='ema'):
    """
    Calculate the Absolute Price Oscillator (APO).

    Parameters:
        data (pd.DataFrame): DataFrame with the closing prices.
        fastperiod (int): The period for the fast EMA. Default is 12.
        slowperiod (int): The period for the slow EMA. Default is 26.
        matype (str): The type of moving average ('ema'). Default is 'ema'.

    Returns:
        pd.Series: A pandas Series representing the APO values.
    """
    data = solve_case(data)
    fast_ma = data['Close'].ewm(span=fastperiod).mean()
    slow_ma = data['Close'].ewm(span=slowperiod).mean()
    apo = fast_ma - slow_ma
    return apo

def AROON(data, period=14):
    """
    Calculate the Aroon indicator.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating Aroon. Default is 14.

    Returns:
        tuple: A tuple containing two pandas Series: Aroon Up and Aroon Down.
    """
    data = solve_case(data)
    aroon_up = ((period - data['High'].rolling(window=period).apply(lambda x: x.argmax())) / period) * 100
    aroon_down = ((period - data['Low'].rolling(window=period).apply(lambda x: x.argmin())) / period) * 100
    return aroon_up, aroon_down

def AROONOSC(data, period=14):
    """
    Calculate the Aroon Oscillator.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating Aroon Oscillator. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the Aroon Oscillator values.
    """
    data = solve_case(data)
    aroon_up, aroon_down = AROON(data['High'], data['Low'], period)
    aroonosc = aroon_up - aroon_down
    return aroonosc

def BOP(data):
    """
    Calculate the Balance Of Power (BOP).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.

    Returns:
        pd.Series: A pandas Series representing the BOP values.
    """
    data = solve_case(data)
    bop = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
    return bop

def CCI(data, period=20):
    """
    Calculate the Commodity Channel Index (CCI).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the CCI. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the CCI values.
    """
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
    cci = (tp - sma) / (0.015 * mad)
    return cci

def CMO(data, period=14):
    """
    Calculate the Chande Momentum Oscillator (CMO).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the CMO. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the CMO values.
    """
    data = solve_case(data)
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    sum_up = up.rolling(window=period).sum()
    sum_down = down.rolling(window=period).sum()
    cmo = 100 * ((sum_up - sum_down) / (sum_up + sum_down))
    return cmo

def COP(data, short_period=11, long_period=14, wma_period=10):
    """
    Calculate the Coppock Curve indicator for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        short_period (int): The short ROC period. Default is 11.
        long_period (int): The long ROC period. Default is 14.
        wma_period (int): The weighted moving average period. Default is 10.

    Returns:
        pd.Series: A pandas Series representing the Coppock Curve values.
    """
    data = solve_case(data)
    roc_short = data['Close'].pct_change(short_period) * 100
    roc_long = data['Close'].pct_change(long_period) * 100
    coppock = roc_short + roc_long
    coppock_curve = coppock.rolling(window=wma_period).mean()
    return pd.Series(coppock_curve, name='Coppock_Curve')

def DX(data, period=14):
    """
    Calculate the Directional Movement Index (DX).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the DX. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the DX values.
    """
    data = solve_case(data)
    plus_dm = PLUS_DM(data['High'], data['Low'])
    minus_dm = MINUS_DM(data['High'], data['Low'])
    tr = TR(data['High'], data['Low'], data['Close'])
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
    return dx

def MACD(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        fast_period (int): The period for the fast EMA. Default is 12.
        slow_period (int): The period for the slow EMA. Default is 26.
        signal_period (int): The period for the signal line. Default is 9.

    Returns:
        pd.DataFrame: A DataFrame with 'MACD', 'Signal_Line', and 'MACD_Histogram' columns.
    """
    data = solve_case(data)
    # Calculate the fast and slow EMAs
    fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd = fast_ema - slow_ema
    
    # Calculate the signal line
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    macd_histogram = macd - signal_line
    
    # Return as DataFrame
    return pd.DataFrame({
        'MACD': macd,
        'Signal_Line': signal_line,
        'MACD_Histogram': macd_histogram
    })

def MACDEXT(data, fastperiod=12, slowperiod=26, signalperiod=9, matype='ema'):
    """
    Calculate the MACD with controllable MA type.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        fastperiod (int): The period for the fast EMA. Default is 12.
        slowperiod (int): The period for the slow EMA. Default is 26.
        signalperiod (int): The period for the signal line. Default is 9.
        matype (str): The type of moving average ('ema'). Default is 'ema'.

    Returns:
        tuple: A tuple containing three pandas Series: MACD Line, Signal Line, and MACD Histogram.
    """
    data = solve_case(data)
    fast_ma = data['Close'].ewm(span=fastperiod).mean()
    slow_ma = data['Close'].ewm(span=slowperiod).mean()
    macd_line = fast_ma - slow_ma
    signal_line = macd_line.ewm(span=signalperiod).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def MACDFIX(close):
    """
    Calculate the MACD Fix 12/26.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.

    Returns:
        tuple: A tuple containing two pandas Series: MACD Line and Signal Line.
    """
    macd_line, signal_line, _ = MACDEXT(close)
    return macd_line, signal_line

def MFI(data, period=14):
    """
    Calculate the Money Flow Index (MFI).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the MFI. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the MFI values.
    """
    data = solve_case(data)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow[typical_price > typical_price.shift(1)].rolling(window=period).sum()
    negative_flow = money_flow[typical_price < typical_price.shift(1)].rolling(window=period).sum()
    mfi = 100 * positive_flow / (positive_flow + negative_flow)
    return mfi

def MINUS_DI(data, period=14):
    """
    Calculate the Minus Directional Indicator (MINUS_DI).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the MINUS_DI. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the MINUS_DI values.
    """
    data = solve_case(data)
    minus_dm = MINUS_DM(data['Low'])
    tr = TR(data['High'], data['Low'], data['Close'])
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    return minus_di

def MINUS_DM(data):
    """
    Calculate the Minus Directional Movement (MINUS_DM).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.

    Returns:
        pd.Series: A pandas Series representing the MINUS_DM values.
    """
    data = solve_case(data)
    minus_dm = data['Low'].diff().clip(lower=0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)
    return minus_dm

def MOM(data, period=10):
    """
    Calculate the Momentum indicator for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        period (int): The number of periods to calculate Momentum. Default is 10.

    Returns:
        pd.Series: A pandas Series representing the Momentum values.
    """
    momentum = data['Close'].diff(period)
    return pd.Series(momentum, name='Momentum')

def PLUS_DI(data, period=14):
    """
    Calculate the Plus Directional Indicator (PLUS_DI).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the PLUS_DI. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the PLUS_DI values.
    """
    data = solve_case(data)
    plus_dm = PLUS_DM(data['High'], data['Low'])
    tr = TR(data['High'], data['Low'], data['Close'])
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    return plus_di

def PLUS_DM(data):
    """
    Calculate the Plus Directional Movement (PLUS_DM).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.

    Returns:
        pd.Series: A pandas Series representing the PLUS_DM values.
    """
    data = solve_case(data)
    plus_dm = data['High'].diff().clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    return plus_dm

def PPO(data, fastperiod=12, slowperiod=26):
    """
    Calculate the Percentage Price Oscillator (PPO).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        fastperiod (int): The period for the fast EMA. Default is 12.
        slowperiod (int): The period for the slow EMA. Default is 26.

    Returns:
        pd.Series: A pandas Series representing the PPO values.
    """
    data = solve_case(data)
    fast_ma = data['Close'].ewm(span=fastperiod).mean()
    slow_ma = data['Close'].ewm(span=slowperiod).mean()
    ppo = (fast_ma - slow_ma) / slow_ma * 100
    return ppo

def ROC(data, period=14):
    """
    Calculate the Rate of Change (ROC).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the ROC. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROC values.
    """
    data = solve_case(data)
    roc = (data['Close'] / data['Close'].shift(period) - 1) * 100
    return roc

def ROCP(data, period=14):
    """
    Calculate the Rate of Change Percentage (ROCP).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the ROCP. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROCP values.
    """
    data = solve_case(data)
    rocp = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
    return rocp

def ROCR(data, period=14):
    """
    Calculate the Rate of Change Ratio (ROCR).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the ROCR. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROCR values.
    """
    data = solve_case(data)
    rocr = data['Close'] / data['Close'].shift(period)
    return rocr

def ROCR100(data, period=14):
    """
    Calculate the Rate of Change Ratio 100 Scale (ROCR100).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the ROCR100. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROCR100 values.
    """
    data = solve_case(data)
    rocr100 = (data['Close'] / data['Close'].shift(period)) * 100
    return rocr100

def RSI(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        period (int): The number of periods to calculate RSI. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the RSI values.
    """
    data = solve_case(data)
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, name='RSI')

def STOCH(data, period=14):
    """
    Calculate the Stochastic Oscillator (STOCH).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the STOCH. Default is 14.

    Returns:
        tuple: A tuple containing two pandas Series: %K and %D.
    """
    data = solve_case(data)
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d

def STOCHRSI(data, period=14):
    """
    Calculate the Stochastic Relative Strength Index (STOCHRSI).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the STOCHRSI. Default is 14.

    Returns:
        tuple: A tuple containing two pandas Series: %K and %D.
    """
    data = solve_case(data)
    rsi = RSI(data, period)
    stoch_rsi_k = 100 * (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean()
    return stoch_rsi_k, stoch_rsi_d

def TR(data):
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift())
        tr3 = abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

def TRIX(data, period=15):
    """
    Calculate the TRIX (Triple Exponential Average).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the TRIX. Default is 15.

    Returns:
        pd.Series: A pandas Series representing the TRIX values.
    """
    data = solve_case(data)
    ema1 = data['Close'].ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()
    trix = ema3.pct_change() * 100
    return trix

def ULTOSC(data, period1=7, period2=14, period3=28):
    """
    Calculate the Ultimate Oscillator (ULTOSC).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period1 (int): The first period for the ULTOSC. Default is 7.
        period2 (int): The second period for the ULTOSC. Default is 14.
        period3 (int): The third period for the ULTOSC. Default is 28.

    Returns:
        pd.Series: A pandas Series representing the ULTOSC values.
    """
    data = solve_case(data)
    
    bp1 = data['Close'] - data['Low'].rolling(window=period1).min()
    tr1 = data['High'].rolling(window=period1).max() - data['Low'].rolling(window=period1).min()
    fp1 = bp1 / tr1

    bp2 = data['Close'] - data['Low'].rolling(window=period2).min()
    tr2 = data['High'].rolling(window=period2).max() - data['Low'].rolling(window=period2).min()
    fp2 = bp2 / tr2

    bp3 = data['Close'] - data['Low'].rolling(window=period3).min()
    tr3 = data['High'].rolling(window=period3).max() - data['Low'].rolling(window=period3).min()
    fp3 = bp3 / tr3

    ultosc = 100 * ((fp1.rolling(window=period1).mean() * 4) + (fp2.rolling(window=period2).mean() * 2) + (fp3.rolling(window=period3).mean())) / 7
    return ultosc

def WILLR(data, period=14):
    """
    Calculate Williams' %R (WILLR).

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' and 'Volume' columns.
        period (int): The period for calculating the WILLR. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the WILLR values.
    """
    data = solve_case(data)
    willr = -100 * ((data['High'].rolling(window=period).max() - data['Close']) / (data['High'].rolling(window=period).max() - data['Low'].rolling(window=period).min()))
    return willr
