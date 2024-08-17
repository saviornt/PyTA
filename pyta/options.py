import pandas as pd
import numpy as np
from scipy.stats import norm

def DELTA(price: float, strike: float, time_to_expiry: float, risk_free_rate: float, 
          volatility: float, is_call: bool=True) -> float:
    """
    Calculate Delta of an option.

    Delta measures the rate of change of the option's price with respect to changes in the underlying asset's price.

    Parameters:
    - price (float): The current price of the underlying asset.
    - strike (float): The strike price of the option.
    - time_to_expiry (float): The time to expiry in years.
    - risk_free_rate (float): The risk-free interest rate.
    - volatility (float): The implied volatility of the option.
    - is_call (bool): Whether the option is a call option (True) or a put option (False).

    Returns:
    - float: The Delta of the option.
    """
    d1 = (np.log(price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1

def GAMMA(price: float, strike: float, time_to_expiry: float, risk_free_rate: float, 
          volatility: float) -> float:
    """
    Calculate Gamma of an option.

    Gamma measures the rate of change of Delta with respect to changes in the underlying asset's price.

    Parameters:
    - price (float): The current price of the underlying asset.
    - strike (float): The strike price of the option.
    - time_to_expiry (float): The time to expiry in years.
    - risk_free_rate (float): The risk-free interest rate.
    - volatility (float): The implied volatility of the option.

    Returns:
    - float: The Gamma of the option.
    """
    d1 = (np.log(price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.pdf(d1) / (price * volatility * np.sqrt(time_to_expiry))

def HV(prices: np.ndarray, window: int=252) -> float:
    """
    Calculate Historical Volatility (HV) over a given window.

    Historical Volatility is the annualized standard deviation of the asset's returns over a specific period.

    Parameters:
    - prices (np.ndarray): Array of historical prices.
    - window (int): The time window for calculating historical volatility, typically 252 for 1 year.

    Returns:
    - float: The Historical Volatility as a percentage.
    """
    log_returns = np.log(prices[1:] / prices[:-1])
    volatility = np.std(log_returns) * np.sqrt(window)
    return volatility * 100  # Return as percentage

def IVBINOMIAL(price: float, strike: float, time_to_expiry: float,
                                risk_free_rate: float, option_price: float,
                                is_call: bool=True, steps: int=100, max_iterations: int=100, 
                                tol: float=1e-5) -> float:
    """
    Calculate the Implied Volatility (IV) using the Binomial model.

    This function uses an iterative approach to estimate the implied volatility based on a binomial tree.

    Parameters:
    - price (float): The current price of the underlying asset.
    - strike (float): The strike price of the option.
    - time_to_expiry (float): The time to expiry in years.
    - risk_free_rate (float): The risk-free interest rate.
    - option_price (float): The current market price of the option.
    - is_call (bool): Whether the option is a call option (True) or a put option (False).
    - steps (int): The number of steps in the binomial tree.
    - max_iterations (int): The maximum number of iterations for convergence.
    - tol (float): The tolerance level for the convergence of the solution.

    Returns:
    - float: The implied volatility as a percentage.
    """
    def binomial_option_price(sigma):
        dt = time_to_expiry / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(risk_free_rate * dt) - d) / (u - d)
        
        option_values = np.zeros(steps + 1)
        for i in range(steps + 1):
            st = price * (u ** (steps - i)) * (d ** i)
            option_values[i] = max(0, (st - strike) if is_call else (strike - st))
        
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-risk_free_rate * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
        
        return option_values[0]
    
    sigma = 0.2  # Initial guess
    for i in range(max_iterations):
        price_estimate = binomial_option_price(sigma)
        if abs(price_estimate - option_price) < tol:
            break
        sigma += 0.001 * np.sign(option_price - price_estimate)
    
    return sigma * 100  # Return as percentage

def IVBLACKSCHOLES(price: float, strike: float, time_to_expiry: float,
                                     risk_free_rate: float, option_price: float, 
                                     is_call: bool=True, max_iterations: int=100, tol: float=1e-5) -> float:
    """
    Calculate the Implied Volatility (IV) using the Black-Scholes model. This model cannot accurately calculate American options since it only
    considers the price at an option's expiration date.

    This function uses an iterative approach (Newton-Raphson) to estimate the implied volatility.

    Parameters:
    - price (float): The current price of the underlying asset.
    - strike (float): The strike price of the option.
    - time_to_expiry (float): The time to expiry in years (e.g., 0.5 for 6 months).
    - risk_free_rate (float): The risk-free interest rate.
    - option_price (float): The current market price of the option.
    - is_call (bool): Whether the option is a call option (True) or a put option (False).
    - max_iterations (int): The maximum number of iterations for convergence.
    - tol (float): The tolerance level for the convergence of the solution.

    Returns:
    - float: The implied volatility as a percentage.
    """
    def black_scholes_price(sigma):
        d1 = (np.log(price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
        d2 = d1 - sigma * np.sqrt(time_to_expiry)
        if is_call:
            return price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            return strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - price * norm.cdf(-d1)
    
    sigma = 0.2  # Initial guess
    for i in range(max_iterations):
        price_estimate = black_scholes_price(sigma)
        vega = price * norm.pdf((np.log(price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))) * np.sqrt(time_to_expiry)
        
        if vega == 0:
            break
        
        sigma -= (price_estimate - option_price) / vega
        if abs(price_estimate - option_price) < tol:
            break
    
    return sigma * 100  # Return as percentage

def PCR(put_volume: float, call_volume: float) -> float:
    """
    Calculate the Put-Call Ratio (PCR).

    The Put-Call Ratio is calculated by dividing the total trading volume of put options 
    by the total trading volume of call options.

    Parameters:
    - put_volume (float): The total volume of put options traded.
    - call_volume (float): The total volume of call options traded.

    Returns:
    - float: The Put-Call Ratio. A value above 1 indicates bearish sentiment, 
             while below 1 suggests bullish sentiment.
    """
    if call_volume == 0:
        raise ValueError("Call volume cannot be zero.")
    
    return put_volume / call_volume

def RHO(price: float, strike: float, time_to_expiry: float, risk_free_rate: float, 
        volatility: float, is_call: bool=True) -> float:
    """
    Calculate Rho of an option.

    Rho measures the rate of change of the option's price with respect to changes in the risk-free interest rate.

    Parameters:
    - price (float): The current price of the underlying asset.
    - strike (float): The strike price of the option.
    - time_to_expiry (float): The time to expiry in years.
    - risk_free_rate (float): The risk-free interest rate.
    - volatility (float): The implied volatility of the option.
    - is_call (bool): Whether the option is a call option (True) or a put option (False).

    Returns:
    - float: The Rho of the option.
    """
    d2 = (np.log(price / strike) + (risk_free_rate - 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    
    if is_call:
        return strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) * 0.01  # Per 1% change in rates
    else:
        return -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) * 0.01


def THETA(price: float, strike: float, time_to_expiry: float, risk_free_rate: float, 
          volatility: float, is_call: bool=True) -> float:
    """
    Calculate Theta of an option.

    Theta measures the rate of change of the option's price with respect to the passage of time.

    Parameters:
    - price (float): The current price of the underlying asset.
    - strike (float): The strike price of the option.
    - time_to_expiry (float): The time to expiry in years.
    - risk_free_rate (float): The risk-free interest rate.
    - volatility (float): The implied volatility of the option.
    - is_call (bool): Whether the option is a call option (True) or a put option (False).

    Returns:
    - float: The Theta of the option.
    """
    d1 = (np.log(price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    first_term = -(price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))
    
    if is_call:
        second_term = risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        second_term = -risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    
    return (first_term + second_term) / 365  # Convert to daily decay

def VEGA(price: float, strike: float, time_to_expiry: float, risk_free_rate: float, 
         volatility: float) -> float:
    """
    Calculate Vega of an option.

    Vega measures the rate of change of the option's price with respect to changes in the volatility of the underlying asset.

    Parameters:
    - price (float): The current price of the underlying asset.
    - strike (float): The strike price of the option.
    - time_to_expiry (float): The time to expiry in years.
    - risk_free_rate (float): The risk-free interest rate.
    - volatility (float): The implied volatility of the option.

    Returns:
    - float: The Vega of the option.
    """
    d1 = (np.log(price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return price * norm.pdf(d1) * np.sqrt(time_to_expiry) * 0.01  # Vega is usually reported per 1% change in volatility


def VS(ivs: dict, strikes: list, atm_strike: float) -> float:
    """
    Calculate the Volatility Skew.

    Volatility Skew is calculated as the difference between the implied volatility of out-of-the-money (OTM)
    options and at-the-money (ATM) options, usually focusing on a specific strike range.

    Parameters:
    - ivs (dict): A dictionary with strike prices as keys and their corresponding implied volatilities as values.
    - strikes (list): A list of strike prices, where you want to calculate the skew (e.g., OTM strikes).
    - atm_strike (float): The strike price that is considered at-the-money (ATM).

    Returns:
    - float: The Volatility Skew as a percentage, indicating the difference between OTM and ATM volatilities.
    """
    if atm_strike not in ivs:
        raise ValueError("ATM strike not found in IV data.")
    
    atm_iv = ivs[atm_strike]
    otm_ivs = [ivs[strike] for strike in strikes if strike in ivs]
    
    if not otm_ivs:
        raise ValueError("No OTM strikes found in IV data.")
    
    average_otm_iv = np.mean(otm_ivs)
    skew = average_otm_iv - atm_iv
    
    return skew * 100  # Return as percentage
