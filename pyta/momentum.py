import pandas as pd
import numpy as np

def ADX(data, period=14):
    """
    Calculate the Average Directional Index (ADX) for a given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
        period (int): The number of periods to calculate ADX. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ADX values.
    """
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

def ADXR(high, low, close, period=14):
    """
    Calculate the Average Directional Movement Index Rating (ADXR).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period (int): The period for calculating the ADX. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ADXR values.
    """
    adx = ADX(high, low, close, period)
    adxr = (adx + adx.shift(period)) / 2
    return adxr

def APO(close, fastperiod=12, slowperiod=26, matype='ema'):
    """
    Calculate the Absolute Price Oscillator (APO).

    Parameters:
        close (pd.Series): The closing prices.
        fastperiod (int): The period for the fast EMA. Default is 12.
        slowperiod (int): The period for the slow EMA. Default is 26.
        matype (str): The type of moving average ('ema'). Default is 'ema'.

    Returns:
        pd.Series: A pandas Series representing the APO values.
    """
    fast_ma = close.ewm(span=fastperiod).mean()
    slow_ma = close.ewm(span=slowperiod).mean()
    apo = fast_ma - slow_ma
    return apo

def AROON(high, low, period=14):
    """
    Calculate the Aroon indicator.

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        period (int): The period for calculating Aroon. Default is 14.

    Returns:
        tuple: A tuple containing two pandas Series: Aroon Up and Aroon Down.
    """
    aroon_up = ((period - high.rolling(window=period).apply(lambda x: x.argmax())) / period) * 100
    aroon_down = ((period - low.rolling(window=period).apply(lambda x: x.argmin())) / period) * 100
    return aroon_up, aroon_down

def AROONOSC(high, low, period=14):
    """
    Calculate the Aroon Oscillator.

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        period (int): The period for calculating Aroon Oscillator. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the Aroon Oscillator values.
    """
    aroon_up, aroon_down = AROON(high, low, period)
    aroonosc = aroon_up - aroon_down
    return aroonosc

def BOP(open_, high, low, close):
    """
    Calculate the Balance Of Power (BOP).

    Parameters:
        open_ (pd.Series): The opening prices.
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.

    Returns:
        pd.Series: A pandas Series representing the BOP values.
    """
    bop = (close - open_) / (high - low)
    return bop

def CCI(high, low, close, period=20):
    """
    Calculate the Commodity Channel Index (CCI).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period (int): The period for calculating the CCI. Default is 20.

    Returns:
        pd.Series: A pandas Series representing the CCI values.
    """
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
    cci = (tp - sma) / (0.015 * mad)
    return cci

def CMO(close, period=14):
    """
    Calculate the Chande Momentum Oscillator (CMO).

    Parameters:
        close (pd.Series): The closing prices.
        period (int): The period for calculating the CMO. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the CMO values.
    """
    delta = close.diff()
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
    roc_short = data['Close'].pct_change(short_period) * 100
    roc_long = data['Close'].pct_change(long_period) * 100
    coppock = roc_short + roc_long
    coppock_curve = coppock.rolling(window=wma_period).mean()
    return pd.Series(coppock_curve, name='Coppock_Curve')

def DX(high, low, close, period=14):
    """
    Calculate the Directional Movement Index (DX).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period (int): The period for calculating the DX. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the DX values.
    """
    plus_dm = PLUS_DM(high, low)
    minus_dm = MINUS_DM(high, low)
    tr = TR(high, low, close)
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

def MACDEXT(close, fastperiod=12, slowperiod=26, signalperiod=9, matype='ema'):
    """
    Calculate the MACD with controllable MA type.

    Parameters:
        close (pd.Series): The closing prices.
        fastperiod (int): The period for the fast EMA. Default is 12.
        slowperiod (int): The period for the slow EMA. Default is 26.
        signalperiod (int): The period for the signal line. Default is 9.
        matype (str): The type of moving average ('ema'). Default is 'ema'.

    Returns:
        tuple: A tuple containing three pandas Series: MACD Line, Signal Line, and MACD Histogram.
    """
    fast_ma = close.ewm(span=fastperiod).mean()
    slow_ma = close.ewm(span=slowperiod).mean()
    macd_line = fast_ma - slow_ma
    signal_line = macd_line.ewm(span=signalperiod).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def MACDFIX(close):
    """
    Calculate the MACD Fix 12/26.

    Parameters:
        close (pd.Series): The closing prices.

    Returns:
        tuple: A tuple containing two pandas Series: MACD Line and Signal Line.
    """
    macd_line, signal_line, _ = MACDEXT(close)
    return macd_line, signal_line

def MFI(high, low, close, volume, period=14):
    """
    Calculate the Money Flow Index (MFI).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        volume (pd.Series): The volume.
        period (int): The period for calculating the MFI. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the MFI values.
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow[typical_price > typical_price.shift(1)].rolling(window=period).sum()
    negative_flow = money_flow[typical_price < typical_price.shift(1)].rolling(window=period).sum()
    mfi = 100 * positive_flow / (positive_flow + negative_flow)
    return mfi

def MINUS_DI(high, low, close, period=14):
    """
    Calculate the Minus Directional Indicator (MINUS_DI).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period (int): The period for calculating the MINUS_DI. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the MINUS_DI values.
    """
    minus_dm = MINUS_DM(low)
    tr = TR(high, low, close)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    return minus_di

def MINUS_DM(low):
    """
    Calculate the Minus Directional Movement (MINUS_DM).

    Parameters:
        low (pd.Series): The low prices.

    Returns:
        pd.Series: A pandas Series representing the MINUS_DM values.
    """
    minus_dm = low.diff().clip(lower=0)
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

def PLUS_DI(high, low, close, period=14):
    """
    Calculate the Plus Directional Indicator (PLUS_DI).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period (int): The period for calculating the PLUS_DI. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the PLUS_DI values.
    """
    plus_dm = PLUS_DM(high, low)
    tr = TR(high, low, close)
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    return plus_di

def PLUS_DM(high, low):
    """
    Calculate the Plus Directional Movement (PLUS_DM).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.

    Returns:
        pd.Series: A pandas Series representing the PLUS_DM values.
    """
    plus_dm = high.diff().clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    return plus_dm

def PPO(close, fastperiod=12, slowperiod=26):
    """
    Calculate the Percentage Price Oscillator (PPO).

    Parameters:
        close (pd.Series): The closing prices.
        fastperiod (int): The period for the fast EMA. Default is 12.
        slowperiod (int): The period for the slow EMA. Default is 26.

    Returns:
        pd.Series: A pandas Series representing the PPO values.
    """
    fast_ma = close.ewm(span=fastperiod).mean()
    slow_ma = close.ewm(span=slowperiod).mean()
    ppo = (fast_ma - slow_ma) / slow_ma * 100
    return ppo

def ROC(close, period=14):
    """
    Calculate the Rate of Change (ROC).

    Parameters:
        close (pd.Series): The closing prices.
        period (int): The period for calculating the ROC. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROC values.
    """
    roc = (close / close.shift(period) - 1) * 100
    return roc

def ROCP(close, period=14):
    """
    Calculate the Rate of Change Percentage (ROCP).

    Parameters:
        close (pd.Series): The closing prices.
        period (int): The period for calculating the ROCP. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROCP values.
    """
    rocp = (close - close.shift(period)) / close.shift(period)
    return rocp

def ROCR(close, period=14):
    """
    Calculate the Rate of Change Ratio (ROCR).

    Parameters:
        close (pd.Series): The closing prices.
        period (int): The period for calculating the ROCR. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROCR values.
    """
    rocr = close / close.shift(period)
    return rocr

def ROCR100(close, period=14):
    """
    Calculate the Rate of Change Ratio 100 Scale (ROCR100).

    Parameters:
        close (pd.Series): The closing prices.
        period (int): The period for calculating the ROCR100. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the ROCR100 values.
    """
    rocr100 = (close / close.shift(period)) * 100
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
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, name='RSI')

def STOCH(high, low, close, period=14):
    """
    Calculate the Stochastic Oscillator (STOCH).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period (int): The period for calculating the STOCH. Default is 14.

    Returns:
        tuple: A tuple containing two pandas Series: %K and %D.
    """
    low_min = low.rolling(window=period).min()
    high_max = high.rolling(window=period).max()
    stoch_k = 100 * (close - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d

def STOCHRSI(close, period=14):
    """
    Calculate the Stochastic Relative Strength Index (STOCHRSI).

    Parameters:
        close (pd.Series): The closing prices.
        period (int): The period for calculating the STOCHRSI. Default is 14.

    Returns:
        tuple: A tuple containing two pandas Series: %K and %D.
    """
    rsi = RSI(close, period)
    stoch_rsi_k = 100 * (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean()
    return stoch_rsi_k, stoch_rsi_d

def TR(high, low, close):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

def TRIX(close, period=15):
    """
    Calculate the TRIX (Triple Exponential Average).

    Parameters:
        close (pd.Series): The closing prices.
        period (int): The period for calculating the TRIX. Default is 15.

    Returns:
        pd.Series: A pandas Series representing the TRIX values.
    """
    ema1 = close.ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()
    trix = ema3.pct_change() * 100
    return trix

def ULTOSC(high, low, close, period1=7, period2=14, period3=28):
    """
    Calculate the Ultimate Oscillator (ULTOSC).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period1 (int): The first period for the ULTOSC. Default is 7.
        period2 (int): The second period for the ULTOSC. Default is 14.
        period3 (int): The third period for the ULTOSC. Default is 28.

    Returns:
        pd.Series: A pandas Series representing the ULTOSC values.
    """
    bp1 = close - low.rolling(window=period1).min()
    tr1 = high.rolling(window=period1).max() - low.rolling(window=period1).min()
    fp1 = bp1 / tr1

    bp2 = close - low.rolling(window=period2).min()
    tr2 = high.rolling(window=period2).max() - low.rolling(window=period2).min()
    fp2 = bp2 / tr2

    bp3 = close - low.rolling(window=period3).min()
    tr3 = high.rolling(window=period3).max() - low.rolling(window=period3).min()
    fp3 = bp3 / tr3

    ultosc = 100 * ((fp1.rolling(window=period1).mean() * 4) + (fp2.rolling(window=period2).mean() * 2) + (fp3.rolling(window=period3).mean())) / 7
    return ultosc

def WILLR(high, low, close, period=14):
    """
    Calculate Williams' %R (WILLR).

    Parameters:
        high (pd.Series): The high prices.
        low (pd.Series): The low prices.
        close (pd.Series): The closing prices.
        period (int): The period for calculating the WILLR. Default is 14.

    Returns:
        pd.Series: A pandas Series representing the WILLR values.
    """
    willr = -100 * ((high.rolling(window=period).max() - close) / (high.rolling(window=period).max() - low.rolling(window=period).min()))
    return willr
