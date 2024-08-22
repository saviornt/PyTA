import pandas as pd
import numpy as np
from column_case_solver import solve_case

def CDL2CROWS(data):
    """
    Two Crows: A bearish reversal pattern that consists of two black (or red) candlesticks following a trend.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Two Crows pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Close'] < data['Close'].shift(1)) &
        (data['Open'] > data['Close'].shift(1)) &
        (data['Close'] < data['Open'].shift(1))
    ).astype(int)

def CDL3BLACKCROWS(data):
    """
    Three Black Crows: A bearish reversal pattern consisting of three consecutive black (or red) candlesticks.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Three Black Crows pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'] < data['Open']) &
        (data['Close'] < data['Close'].shift(1)) &
        (data['Open'] > data['Close'].shift(1)) &
        (data['Close'] < data['Close'].shift(2))
    ).astype(int)

def CDL3INSIDE(data):
    """
    Three Inside Up/Down: A reversal pattern where a smaller candle is engulfed by a larger candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Three Inside Up/Down pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'].shift(1) > data['Close'].shift(1)) &
        (data['Close'] > data['Open'].shift(2)) &
        (data['Open'] < data['Close'].shift(2)) &
        (data['Close'] > data['Open']) &
        (data['Open'] < data['Close'])
    ).astype(int)

def CDL3LINESTRIKE(data):
    """
    Three-Line Strike: A bullish or bearish reversal pattern characterized by three consecutive candles followed by a fourth candle that negates the previous three.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Three-Line Strike pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(3) < data['Open'].shift(3)) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] < data['Open'].shift(3)) &
        (data['Open'] > data['Close'].shift(1)) &
        (data['Close'] < data['Open'])
    ).astype(int)

def CDL3OUTSIDE(data):
    """
    Three Outside Up/Down: A reversal pattern consisting of a large candle that engulfs the previous two candles.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Three Outside Up/Down pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'].shift(1) > data['Close'].shift(1)) &
        (data['Close'] > data['Open'].shift(2)) &
        (data['Open'] < data['Close'].shift(2)) &
        (data['Close'] > data['Open']) &
        (data['Open'] < data['Close'])
    ).astype(int)

def CDL3STARSINSOUTH(data):
    """
    Three Stars In The South: A bearish reversal pattern characterized by three small candles forming a pattern of three stars.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Three Stars In The South pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Close'] < data['Close'].shift(1)) &
        (data['Open'] > data['Close'].shift(1)) &
        (data['Close'] < data['Open'].shift(1))
    ).astype(int)

def CDL3WHITESOLDIERS(data):
    """
    Three Advancing White Soldiers: A bullish reversal pattern consisting of three consecutive long white (or green) candlesticks.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Three Advancing White Soldiers pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'].shift(2) > data['Open'].shift(2)) &
        (data['Close'] > data['Open']) &
        (data['Close'] > data['Close'].shift(1)) &
        (data['Open'] < data['Close'].shift(1)) &
        (data['Close'] > data['Open'].shift(2))
    ).astype(int)

def CDLABANDONEDBABY(data):
    """
    Abandoned Baby: A pattern characterized by a Doji candle between two candles with gaps.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Abandoned Baby pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'].shift(1) > data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'].shift(1) > data['Close'].shift(1))
    ).astype(int)

def CDLADVANCEBLOCK(data):
    """
    Advance Block: A bearish pattern consisting of three white (or green) candlesticks followed by a black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Advance Block pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'].shift(1) > data['Close']) &
        (data['Close'] < data['Open'])
    ).astype(int)

def CDLBELTHOLD(data):
    """
    Belt-hold: A one-candle pattern where a long black (or red) candle follows a bullish trend.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Belt-hold pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] < data['Open']) &
        (data['Close'] < data['Close'].shift(1)) &
        (data['Open'] > data['Close'].shift(1))
    ).astype(int)

def CDLBREAKAWAY(data):
    """
    Breakaway: A pattern consisting of five candlesticks with a gap in the middle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Breakaway pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(4) < data['Open'].shift(4)) &
        (data['Close'].shift(3) < data['Open'].shift(3)) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLCLOSINGMARUBOZU(data):
    """
    Closing Marubozu: A candle with a long body and no shadow.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Closing Marubozu pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'] == data['Low']) &
        (data['Close'] == data['High'])
    ).astype(int)

def CDLCONCEALBABYSWALL(data):
    """
    Concealing Baby Swallow: A bearish pattern characterized by a Doji between two long candlesticks.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Concealing Baby Swallow pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'].shift(1) > data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'].shift(1) > data['Close'].shift(1))
    ).astype(int)

def CDLCOUNTERATTACK(data):
    """
    Counterattack: A bullish or bearish reversal pattern characterized by a reversal of the previous trend.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Counterattack pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'] > data['Open'].shift(1)) &
        (data['Open'] < data['Close'])
    ).astype(int)

def CDLDARKCLOUDCOVER(data):
    """
    Dark Cloud Cover: A bearish reversal pattern with a large white (or green) candle followed by a large black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Dark Cloud Cover pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'] < data['Open'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Close'] < data['Open'])
    ).astype(int)

def CDLDOJI(data):
    """
    Doji: A candle with a very small body and long shadows.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Doji pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] == data['Open']) &
        (data['High'] - data['Close'] < (data['Close'] - data['Open']) * 0.1) &
        (data['Low'] - data['Close'] < (data['Close'] - data['Open']) * 0.1)
    ).astype(int)

def CDLDOJISTAR(data):
    """
    Doji Star: A pattern where a Doji is preceded by a long candle and followed by a long candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Doji Star pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Close'] == data['Open']) &
        (data['Close'].shift(2) > data['Open'].shift(2))
    ).astype(int)

def CDLDRAGONFLYDOJI(data):
    """
    Dragonfly Doji: A Doji with a long lower shadow and a small body.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Dragonfly Doji pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] == data['Open']) &
        (data['Low'] < data['Open']) &
        (data['High'] - data['Close'] < (data['Close'] - data['Open']) * 0.1)
    ).astype(int)

def CDLENGULFING(data):
    """
    Engulfing Pattern: A reversal pattern where a small candle is engulfed by a larger candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Engulfing Pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'].shift(1) < data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Close'] > data['Open'].shift(1)) &
        (data['Open'] < data['Close'].shift(1))
    ).astype(int)

def CDLEVENINGDOJISTAR(data):
    """
    Evening Doji Star: A bearish pattern characterized by a Doji star after a long white (or green) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Evening Doji Star pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Close'] == data['Open']) &
        (data['Close'].shift(2) > data['Open'].shift(2))
    ).astype(int)

def CDLEVENINGSTAR(data):
    """
    Evening Star: A bearish pattern consisting of a long white (or green) candle, followed by a small candle, and then a long black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Evening Star pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'] < data['Open'].shift(1)) &
        (data['Close'].shift(2) < data['Open'].shift(2))
    ).astype(int)

def CDLGAPSIDESIDEWHITE(data):
    """
    Up/Down-gap side-by-side white lines: A pattern consisting of two or more white (or green) candles with gaps.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Up/Down-gap side-by-side white lines pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] < data['Close'].shift(1))
    ).astype(int)

def CDLGRAVESTONEDOJI(data):
    """
    Gravestone Doji: A Doji with a long upper shadow and a small body.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Gravestone Doji pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] == data['Open']) &
        (data['High'] > data['Open']) &
        (data['Low'] > data['Close'])
    ).astype(int)

def CDLHAMMER(data):
    """
    Hammer: A bullish reversal pattern characterized by a small body at the upper end of the trading range with a long lower shadow.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Hammer pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] > data['Open']) &
        (data['Low'] < data['Open']) &
        (data['High'] - data['Close'] < (data['Close'] - data['Open']) * 0.1)
    ).astype(int)

def CDLHANGINGMAN(data):
    """
    Hanging Man: A bearish reversal pattern with a small body and a long lower shadow.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Hanging Man pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] < data['Open']) &
        (data['Low'] < data['Open']) &
        (data['High'] - data['Close'] < (data['Open'] - data['Close']) * 0.1)
    ).astype(int)

def CDLHARAMI(data):
    """
    Harami Pattern: A reversal pattern where a small candle is contained within the body of a larger candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Harami Pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'].shift(1) > data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] < data['Close'].shift(1)) &
        (data['Close'] < data['Open'].shift(1))
    ).astype(int)

def CDLHARAMICROSS(data):
    """
    Harami Cross Pattern: A variation of the Harami Pattern where the small candle is a Doji.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Harami Cross Pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] == data['Open']) &
        (data['Open'] < data['Close'].shift(1)) &
        (data['Close'] < data['Open'].shift(1))
    ).astype(int)

def CDLHIGHWAVE(data):
    """
    High-Wave Candle: A pattern characterized by long upper and lower shadows with a small body.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the High-Wave Candle pattern is present.
    """
    data = solve_case(data)
    return (
        (data['High'] - data['Close'] > (data['Close'] - data['Open']) * 1.5) &
        (data['Open'] - data['Low'] > (data['Close'] - data['Open']) * 1.5)
    ).astype(int)

def CDLHIKKAKE(data):
    """
    Hikkake Pattern: A bullish or bearish pattern characterized by a failed breakout followed by a reversal.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Hikkake Pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'] < data['Close'])
    ).astype(int)

def CDLHIKKAKEMOD(data):
    """
    Modified Hikkake Pattern: A variation of the Hikkake Pattern with different breakout criteria.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Modified Hikkake Pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Open'] < data['Close'])
    ).astype(int)

def CDLHOMINGPIGEON(data):
    """
    Homing Pigeon: A bullish reversal pattern characterized by a Doji followed by a long white (or green) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Homing Pigeon pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] == data['Open']) &
        (data['Close'].shift(2) > data['Open'].shift(2)) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLIDENTICAL3CROWS(data):
    """
    Identical Three Crows: A bearish pattern with three consecutive black (or red) candles of equal size.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Identical Three Crows pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'] < data['Open']) &
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'] == data['Open'].shift(1)) &
        (data['Close'] == data['Open'].shift(2))
    ).astype(int)

def CDLINNECK(data):
    """
    In-Neck Pattern: A bearish pattern where a small candle is followed by a long black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the In-Neck Pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'] < data['Open']) &
        (data['Close'].shift(1) < data['Open'].shift(1))
    ).astype(int)

def CDLINVERTEDHAMMER(data):
    """
    Inverted Hammer: A bullish reversal pattern characterized by a small body and a long upper shadow.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Inverted Hammer pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] > data['Open']) &
        (data['High'] - data['Close'] > (data['Close'] - data['Open']) * 1.5) &
        (data['Open'] - data['Low'] < (data['Close'] - data['Open']) * 0.1)
    ).astype(int)

def CDLKICKING(data):
    """
    Kicking: A pattern where a large white (or green) candle is followed by a large black (or red) candle or vice versa.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Kicking pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'].shift(1) < data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] > data['Close']) &
        (data['Close'].shift(1) < data['Open'].shift(1))
    ).astype(int)

def CDLKICKINGBYLENGTH(data):
    """
    Kicking - bull/bear determined by the longer marubozu: A pattern where a long marubozu is followed by a long marubozu of opposite color.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Kicking - bull/bear determined by the longer marubozu pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'].shift(1) < data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] > data['Close']) &
        (data['Close'].shift(1) < data['Open'].shift(1))
    ).astype(int)

def CDLLADDERBOTTOM(data):
    """
    Ladder Bottom: A bullish reversal pattern characterized by three or more white (or green) candles with small bodies and long lower shadows.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Ladder Bottom pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'] < data['Open'].shift(1)) &
        (data['Open'] > data['Close'].shift(2))
    ).astype(int)

def CDLLONGLEGGEDDOJI(data):
    """
    Long Legged Doji: A Doji with long upper and lower shadows.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Long Legged Doji pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] == data['Open']) &
        (data['High'] - data['Close'] > (data['Close'] - data['Open']) * 1.5) &
        (data['Low'] - data['Close'] > (data['Close'] - data['Open']) * 1.5)
    ).astype(int)

def CDLMARUBOZU(data):
    """
    Marubozu: A candle with no shadow at either end.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Marubozu pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'] == data['Low']) &
        (data['Close'] == data['High'])
    ).astype(int)

def CDLMASTAR(data):
    """
    Mat Hold: A bullish continuation pattern that appears in an uptrend and is characterized by a long white candlestick followed by a series of black candles, and then a breakout above the high of the first white candlestick.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Mat Hold pattern is present.
    """
    data = solve_case(data)
    # Ensure there are at least 5 data points for the pattern
    if len(data) < 5:
        return pd.Series([0] * len(data), index=data.index)

    # Calculate the necessary conditions for the Mat Hold pattern
    first_candle_bullish = data['Close'].shift(4) > data['Open'].shift(4)
    second_candle_bearish = data['Close'].shift(3) < data['Open'].shift(3)
    third_candle_bearish = data['Close'].shift(2) < data['Open'].shift(2)
    fourth_candle_bearish = data['Close'].shift(1) < data['Open'].shift(1)
    fifth_candle_bullish = data['Close'] > data['Open']
    
    breakout = data['High'] > data['High'].shift(4)
    no_breakout = data['High'].shift(1) < data['High'].shift(4)

    return (
        first_candle_bullish &
        second_candle_bearish &
        third_candle_bearish &
        fourth_candle_bearish &
        fifth_candle_bullish &
        breakout &
        no_breakout
    ).astype(int)

def CDLMATHOLD(data):
    """
    Mat Hold: A bullish continuation pattern that appears in an uptrend. It is characterized by a long white candlestick followed by a series of black candles, and then a breakout above the high of the first white candlestick.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Mat Hold pattern is present.
    """
    data = solve_case(data)
    # Ensure there are at least 5 data points for the pattern
    if len(data) < 5:
        return pd.Series([0] * len(data), index=data.index)

    # Define conditions for the Mat Hold pattern
    first_candle_bullish = data['Close'].shift(4) > data['Open'].shift(4)
    second_candle_bearish = data['Close'].shift(3) < data['Open'].shift(3)
    third_candle_bearish = data['Close'].shift(2) < data['Open'].shift(2)
    fourth_candle_bearish = data['Close'].shift(1) < data['Open'].shift(1)
    fifth_candle_bullish = data['Close'] > data['Open']
    
    breakout = data['High'] > data['High'].shift(4)
    no_breakout = data['High'].shift(1) < data['High'].shift(4)

    return (
        first_candle_bullish &
        second_candle_bearish &
        third_candle_bearish &
        fourth_candle_bearish &
        fifth_candle_bullish &
        breakout &
        no_breakout
    ).astype(int)

def CDLMEETINGLINES(data):
    """
    Meeting Lines: A pattern where two consecutive candles have opposite colors and open and close at the same price.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Meeting Lines pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Open'] == data['Close'].shift(1)) &
        (data['Close'] == data['Open'])
    ).astype(int)

def CDLMORNINGDOJISTAR(data):
    """
    Morning Doji Star: A bullish pattern characterized by a Doji star after a long black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Morning Doji Star pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] == data['Open']) &
        (data['Close'].shift(2) > data['Open'].shift(2)) &
        (data['Close'].shift(2) < data['Open'])
    ).astype(int)

def CDLMORNINGSTAR(data):
    """
    Morning Star: A bullish reversal pattern characterized by a long bearish candle, a small-bodied candle, and a long bullish candle, indicating a reversal at the end of a downtrend.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Morning Star pattern is present.
    """
    data = solve_case(data)
    # Ensure there are at least 3 data points for the pattern
    if len(data) < 3:
        return pd.Series([0] * len(data), index=data.index)

    # Calculate the necessary conditions for the Morning Star pattern
    first_candle_bearish = data['Close'].shift(2) < data['Open'].shift(2)
    second_candle_small_body = abs(data['Close'].shift(1) - data['Open'].shift(1)) < (data['High'].shift(1) - data['Low'].shift(1)) * 0.3
    third_candle_bullish = data['Close'] > data['Open']
    gap_down = data['Low'].shift(1) > data['Close'].shift(2)
    gap_up = data['Open'] < data['Close']

    return (
        first_candle_bearish &
        second_candle_small_body &
        third_candle_bullish &
        gap_down &
        gap_up
    ).astype(int)

def CDLONNECK(data):
    """
    On-Neck Pattern: A bearish pattern where a small candle is followed by a long black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the On-Neck Pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'] < data['Open']) &
        (data['Close'].shift(1) < data['Open'].shift(1))
    ).astype(int)

def CDLOPENINGMARUBOZU(data):
    """
    Opening Marubozu: A candle with no shadow on the opening side.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Opening Marubozu pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'] == data['Low']) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLOVERLAPPING(data):
    """
    Overlapping: A pattern where the current candle overlaps with the previous candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Overlapping pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'] < data['Close'].shift(1)) &
        (data['Close'] > data['Open'].shift(1)) &
        (data['Open'] < data['Close'].shift(1))
    ).astype(int)

def CDLPIERCING(data):
    """
    Piercing Pattern: A bullish reversal pattern that occurs in a downtrend. It starts with a long black candlestick followed by a long white candlestick that opens lower than the low of the black candle but closes above the midpoint of the black candle's body.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Piercing Pattern is present.
    """
    data = solve_case(data)
    # Ensure there are at least 2 data points for the pattern
    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    # Define conditions for the Piercing Pattern
    first_candle_black = data['Close'].shift(1) < data['Open'].shift(1)
    second_candle_white = data['Close'] > data['Open']
    
    # The second candle should open below the low of the first candle
    open_below_prev_low = data['Open'] < data['Low'].shift(1)
    
    # The second candle should close above the midpoint of the first candle
    close_above_midpoint = data['Close'] > (data['Open'].shift(1) + data['Close'].shift(1)) / 2

    return (
        first_candle_black &
        second_candle_white &
        open_below_prev_low &
        close_above_midpoint
    ).astype(int)

def CDLPREGNANT(data):
    """
    Pregnant: A bearish pattern where the body of the current candle is contained within the body of the previous candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Pregnant pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'].shift(1) > data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] < data['Close'].shift(1)) &
        (data['Close'] < data['Open'].shift(1))
    ).astype(int)

def CDLRICKSHAWMAN(data):
    """
    Rickshaw Man: A reversal pattern characterized by a candlestick with a small body located in the middle of the trading range with long upper and lower shadows. It indicates indecision in the market and potential reversal.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Rickshaw Man pattern is present.
    """
    data = solve_case(data)
    # Ensure there are enough data points
    if len(data) < 1:
        return pd.Series([0] * len(data), index=data.index)

    # Calculate body size and shadow sizes
    body_size = abs(data['Close'] - data['Open'])
    upper_shadow = data['High'] - data[['Close', 'Open']].max(axis=1)
    lower_shadow = data[['Close', 'Open']].min(axis=1) - data['Low']

    # Define conditions for Rickshaw Man
    is_rickshaw_man = (
        body_size / (upper_shadow + body_size + lower_shadow) < 0.3
        & (upper_shadow > 2 * body_size)
        & (lower_shadow > 2 * body_size)
    )

    return is_rickshaw_man.astype(int)

def CDLRISEFALL3METHODS(data):
    """
    Rising/Falling Three Methods: A continuation pattern characterized by three candles that have small bodies, with a preceding and succeeding long candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Rising/Falling Three Methods pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Open'] < data['Close']) &
        (data['Close'] > data['Open'].shift(1)) &
        (data['Close'].shift(2) < data['Open'].shift(2))
    ).astype(int)

def CDLSEPARATINGLINES(data):
    """
    Separating Lines: A bullish pattern where a long white (or green) candle is followed by a gap up and a long black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Separating Lines pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Open'] < data['Close']) &
        (data['Close'] < data['Open'].shift(1)) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLSHOOTINGSTAR(data):
    """
    Shooting Star: A bearish reversal pattern characterized by a candlestick with a small body at the lower end, a long upper shadow, and little or no lower shadow. It suggests a potential reversal from an uptrend to a downtrend.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Shooting Star pattern is present.
    """
    data = solve_case(data)
    # Ensure there are enough data points
    if len(data) < 1:
        return pd.Series([0] * len(data), index=data.index)

    # Calculate body size, upper shadow, and lower shadow
    body_size = abs(data['Close'] - data['Open'])
    upper_shadow = data['High'] - data[['Close', 'Open']].max(axis=1)
    lower_shadow = data[['Close', 'Open']].min(axis=1) - data['Low']

    # Define conditions for Shooting Star
    is_shooting_star = (
        body_size / (upper_shadow + body_size + lower_shadow) < 0.3
        & (upper_shadow > 2 * body_size)
        & (lower_shadow < 0.1 * body_size)
        & (data['Close'] < data['Open'])
    )

    return is_shooting_star.astype(int)

def CDLSHORTLINE(data):
    """
    Short Line: A pattern characterized by a small body with long shadows.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Short Line pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] > data['Open']) &
        (data['High'] - data['Close'] > (data['Close'] - data['Open']) * 1.5) &
        (data['Open'] - data['Low'] > (data['Close'] - data['Open']) * 1.5)
    ).astype(int)

def CDLSPINNINGTOP(data):
    """
    Spinning Top: A pattern with a small body and long shadows.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Spinning Top pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] > data['Open']) &
        (data['High'] - data['Close'] > (data['Close'] - data['Open']) * 1.5) &
        (data['Open'] - data['Low'] > (data['Close'] - data['Open']) * 1.5)
    ).astype(int)

def CDLSTALLEDPATTERN(data):
    """
    Stalled Pattern: A bearish reversal pattern characterized by a long white (bullish) candlestick followed by a small-bodied candlestick (either white or black) which indicates indecision and potential reversal.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Stalled Pattern is present.
    """
    data = solve_case(data)
    # Ensure there are enough data points
    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    # Shift data to compare current and previous day
    prev = data.shift(1)

    # Calculate body size for current and previous day
    body_size_current = abs(data['Close'] - data['Open'])
    body_size_prev = abs(prev['Close'] - prev['Open'])

    # Define conditions for Stalled Pattern
    is_stalled_pattern = (
        (data['Open'] < data['Close']) &  # Current candle is bullish
        (prev['Open'] < prev['Close']) &  # Previous candle is bullish
        (data['Close'] < prev['Close']) &  # Current close is lower than previous close
        (body_size_current < 0.5 * body_size_prev)  # Current body is smaller than previous
    )

    return is_stalled_pattern.astype(int)

def CDLSTICKSANDWICH(data):
    """
    Sticks Sandwich: A bearish pattern characterized by a large candle with two smaller candles inside its range.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Sticks Sandwich pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Open'] > data['Close'].shift(1)) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLSTICKSWITHIN(data):
    """
    Sticks Within: A bearish pattern where the current candle is contained within the previous candle's range.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Sticks Within pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Open'] < data['Open'].shift(1)) &
        (data['Close'] < data['Close'].shift(1)) &
        (data['Open'] > data['Open'].shift(1)) &
        (data['Close'] > data['Close'].shift(1))
    ).astype(int)

def CDLTAKURI(data):
    """
    Takuri: A bullish pattern with a long lower shadow and a small body.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Takuri pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] > data['Open']) &
        (data['Open'] - data['Low'] > (data['Close'] - data['Open']) * 1.5) &
        (data['High'] - data['Close'] < (data['Close'] - data['Open']) * 0.1)
    ).astype(int)

def CDLTASUKIGAP(data):
    """
    Tasuki Gaps: A pattern where a gap up is followed by a small candle within the gap and then a long candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Tasuki Gaps pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Open'] < data['Close']) &
        (data['Close'] < data['Open'].shift(1)) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLTHRUSTING(data):
    """
    Thrusting: A bullish pattern where a long white (or green) candle is followed by a small black (or red) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Thrusting pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Close'] > data['Open']) &
        (data['Close'].shift(1) < data['Open'].shift(1))
    ).astype(int)

def CDLTRISTAR(data):
    """
    Tri-Star: A rare and reliable reversal pattern with three Doji candles.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Tri-Star pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] == data['Open']) &
        (data['Close'].shift(1) == data['Open'].shift(1)) &
        (data['Close'].shift(2) == data['Open'].shift(2))
    ).astype(int)

def CDLUNIQUE3RIVER(data):
    """
    Unique Three River: A bullish pattern with three candles, the last of which is a long white (or green) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Unique Three River pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'] < data['Open']) &
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'].shift(2) > data['Open'].shift(2)) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLUPSIDEGAP2CROWS(data):
    """
    Upside Gap Two Crows: A bearish pattern with a gap up followed by two black (or red) candles.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Upside Gap Two Crows pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Open'] < data['Close']) &
        (data['Close'] < data['Open'].shift(1)) &
        (data['Close'] > data['Open'])
    ).astype(int)

def CDLVALE(data):
    """
    Vale: A bearish pattern with a long black (or red) candle followed by a small white (or green) candle.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Vale pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] > data['Close'].shift(1)) &
        (data['Close'] < data['Open'])
    ).astype(int)

def CDLVARIETY(data):
    """
    Variety: A bullish pattern characterized by a series of candles with increasing or decreasing bodies.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Variety pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] > data['Open']) &
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'].shift(2) > data['Open'].shift(2))
    ).astype(int)

def CDLWHITESOLDIER(data):
    """
    White Soldier: A bullish pattern characterized by three consecutive white (or green) candles with increasing sizes.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the White Soldier pattern is present.
    """
    data = solve_case(data)
    return (
        (data['Close'] > data['Open']) &
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'].shift(2) > data['Open'].shift(2))
    ).astype(int)

def CDLXSIDEGAP3METHODS(data):
    """
    Upside/Downside Gap Three Methods: A continuation pattern that occurs after a strong trend, characterized by three small-bodied candles that gap up or down from the previous candles, indicating continuation of the trend.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series indicating where the Upside/Downside Gap Three Methods pattern is present.
    """
    data = solve_case(data)
    # Ensure there are enough data points
    if len(data) < 5:
        return pd.Series([0] * len(data), index=data.index)

    # Shift data to compare current and previous days
    prev = data.shift(1)
    prev2 = data.shift(2)
    prev3 = data.shift(3)
    prev4 = data.shift(4)

    # Define conditions for Upside/Downside Gap Three Methods
    gap_up = (prev['Close'] < data['Open']) & (data['Open'] < data['Close']) & (data['Close'] < prev['Open'])
    gap_down = (prev['Close'] > data['Open']) & (data['Open'] > data['Close']) & (data['Close'] > prev['Open'])

    body_size_current = abs(data['Close'] - data['Open'])
    body_size_prev = abs(prev['Close'] - prev['Open'])
    body_size_prev2 = abs(prev2['Close'] - prev2['Open'])
    body_size_prev3 = abs(prev3['Close'] - prev3['Open'])
    body_size_prev4 = abs(prev4['Close'] - prev4['Open'])

    is_xside_gap_3_methods = (
        # Check if the first candle is a large bullish or bearish candle
        (body_size_prev >= 2 * body_size_prev2) &
        # Check for three small candles with gaps
        ((gap_up & (body_size_current < 0.5 * body_size_prev) & (body_size_prev2 < 0.5 * body_size_prev)) |
         (gap_down & (body_size_current < 0.5 * body_size_prev) & (body_size_prev2 < 0.5 * body_size_prev)))
    )

    return is_xside_gap_3_methods.astype(int)
