import numpy as np
import pandas as pd
from scipy.signal import hilbert

def HT_DCPERIOD(data):
    """
    Calculate Hilbert Transform Dominant Cycle Period (HT_DCPERIOD).

    Parameters:
        data (pd.Series): Series containing the price data.

    Returns:
        pd.Series: A pandas Series representing the dominant cycle period.
    """
    analytic_signal = hilbert(data)
    phase = np.angle(analytic_signal)
    instantaneous_frequency = np.diff(phase) / (2.0 * np.pi)
    period = 1 / instantaneous_frequency
    period = np.append(np.nan, period)  # Maintain the original length
    return pd.Series(period, index=data.index)

def HT_DCPHASE(data):
    """
    Calculate Hilbert Transform Dominant Cycle Phase (HT_DCPHASE).

    Parameters:
        data (pd.Series): Series containing the price data.

    Returns:
        pd.Series: A pandas Series representing the dominant cycle phase.
    """
    analytic_signal = hilbert(data)
    phase = np.angle(analytic_signal)
    return pd.Series(phase, index=data.index)

def HT_PHASOR(data):
    """
    Calculate Hilbert Transform Phasor (HT_PHASOR).

    Parameters:
        data (pd.Series): Series containing the price data.

    Returns:
        pd.DataFrame: DataFrame with 'In-Phase' and 'Quadrature' components.
    """
    analytic_signal = hilbert(data)
    in_phase = np.real(analytic_signal)
    quadrature = np.imag(analytic_signal)
    return pd.DataFrame({
        'In-Phase': in_phase,
        'Quadrature': quadrature
    }, index=data.index)

def HT_SINE(data):
    """
    Calculate Hilbert Transform Sine (HT_SINE).

    Parameters:
        data (pd.Series): Series containing the price data.

    Returns:
        pd.Series: A pandas Series representing the sine component of the signal.
    """
    analytic_signal = hilbert(data)
    sine_component = np.imag(analytic_signal)
    return pd.Series(sine_component, index=data.index)

def HT_TRENDMODE(data):
    """
    Calculate Hilbert Transform Trend Mode (HT_TRENDMODE).

    Parameters:
        data (pd.Series): Series containing the price data.

    Returns:
        pd.Series: A pandas Series representing the trend mode (0 for downtrend, 1 for uptrend).
    """
    analytic_signal = hilbert(data)
    in_phase = np.real(analytic_signal)
    trend_mode = np.where(in_phase > 0, 1, 0)
    return pd.Series(trend_mode, index=data.index)
