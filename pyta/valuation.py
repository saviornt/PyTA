import numpy as np
import pandas as pd

def DR(market_value_equity, market_value_debt, cost_of_equity, cost_of_debt, tax_rate):
    """
    Calculate the Weighted Average Cost of Capital (WACC) which is typically used as the discount rate.

    Parameters:
    market_value_equity (float): The market value of the company's equity.
    market_value_debt (float): The market value of the company's debt.
    cost_of_equity (float): The cost of equity (in percentage).
    cost_of_debt (float): The cost of debt (in percentage).
    tax_rate (float): The corporate tax rate (in percentage).

    Returns:
    float: The WACC as a percentage, rounded to two decimal places.
    """
    total_value = market_value_equity + market_value_debt

    weight_equity = market_value_equity / total_value
    weight_debt = market_value_debt / total_value

    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    return np.round(wacc, 2)

def DCF_VALUE(free_cash_flow, discount_rate, terminal_value):
    """
    Calculate the Discounted Cash Flow (DCF) value.

    The DCF value is calculated as the present value of expected future cash flows 
    plus the present value of the terminal value.

    Parameters:
    free_cash_flow (float): The expected future free cash flow.
    discount_rate (float): The discount rate (as a percentage).
    terminal_value (float): The terminal value of the company.

    Returns:
    float: The DCF value, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if free_cash_flow <= 0 or discount_rate <= 0 or terminal_value <= 0:
        raise ValueError("Free cash flow, discount rate, and terminal value must be positive.")

    present_value_fcf = free_cash_flow / (1 + discount_rate)
    present_value_terminal = terminal_value / (1 + discount_rate)

    dcf_value = present_value_fcf + present_value_terminal
    return np.round(dcf_value, 2)

def DDM_VALUE(expected_dividend, dividend_growth_rate, discount_rate):
    """
    Calculate the Dividend Discount Model (DDM) value.

    The DDM value is calculated as the expected dividend per share divided by
    the difference between the discount rate and the dividend growth rate.

    Parameters:
    expected_dividend (float): The expected annual dividend per share.
    dividend_growth_rate (float): The annual growth rate of the dividend (as a percentage).
    discount_rate (float): The discount rate (as a percentage).

    Returns:
    float: The DDM value, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive or if the discount rate is less than or equal to the dividend growth rate.
    """
    if expected_dividend <= 0 or dividend_growth_rate < 0 or discount_rate <= 0:
        raise ValueError("Expected dividend, discount rate, and dividend growth rate must be positive.")

    if discount_rate <= dividend_growth_rate:
        raise ValueError("Discount rate must be greater than the dividend growth rate.")

    ddm_value = expected_dividend / (discount_rate - dividend_growth_rate)
    return np.round(ddm_value, 2)

def DIVIDEND_YIELD(annual_dividend_per_share, price):
    """
    Calculate the Dividend Yield.

    The Dividend Yield is calculated as the annual dividend per share divided by 
    the current price per share.

    Parameters:
    annual_dividend_per_share (float): The annual dividend per share.
    price (float): The price per share of the stock.

    Returns:
    float: The Dividend Yield as a percentage, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if annual_dividend_per_share <= 0 or price <= 0:
        raise ValueError("All input values must be positive.")

    dividend_yield = (annual_dividend_per_share / price) * 100
    return np.round(dividend_yield, 2)

def EARNINGS_YIELD(earnings_per_share, price):
    """
    Calculate the Earnings Yield.

    The Earnings Yield is calculated as the earnings per share (EPS) divided by 
    the current price per share.

    Parameters:
    earnings_per_share (float): The earnings per share of the stock.
    price (float): The price per share of the stock.

    Returns:
    float: The Earnings Yield as a percentage, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if earnings_per_share <= 0 or price <= 0:
        raise ValueError("All input values must be positive.")

    earnings_yield = (earnings_per_share / price) * 100
    return np.round(earnings_yield, 2)

def ENTERPRISE_VALUE(market_cap, total_debt, cash_and_equivalents):
    """
    Calculate the Enterprise Value (EV).

    Enterprise Value is calculated as the market capitalization plus total debt minus 
    cash and equivalents.

    Parameters:
    market_cap (float): The market capitalization of the company.
    total_debt (float): The total debt of the company.
    cash_and_equivalents (float): The cash and equivalents of the company.

    Returns:
    float: The Enterprise Value, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are negative.
    """
    if market_cap < 0 or total_debt < 0 or cash_and_equivalents < 0:
        raise ValueError("Market capitalization, total debt, and cash and equivalents must be non-negative.")

    ev = market_cap + total_debt - cash_and_equivalents
    return np.round(ev, 2)

def EV_TO_EBITDA(ev, ebitda):
    """
    Calculate the Enterprise Value to EBITDA (EV/EBITDA) ratio.

    The EV/EBITDA ratio is calculated as the Enterprise Value divided by 
    the Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA).

    Parameters:
    ev (float): The Enterprise Value.
    ebitda (float): The EBITDA of the company.

    Returns:
    float: The EV/EBITDA ratio, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if ev <= 0 or ebitda <= 0:
        raise ValueError("Enterprise Value and EBITDA must be positive.")

    ev_to_ebitda = ev / ebitda
    return np.round(ev_to_ebitda, 2)

def PB_RATIO(price, book_value_per_share):
    """
    Calculate the Price-to-Book (P/B) ratio.

    The P/B ratio is calculated as the current price per share divided by 
    the book value per share.

    Parameters:
    price (float): The price per share of the stock.
    book_value_per_share (float): The book value per share of the stock.

    Returns:
    float: The P/B ratio, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if price <= 0 or book_value_per_share <= 0:
        raise ValueError("All input values must be positive.")

    pb_ratio = price / book_value_per_share
    return np.round(pb_ratio, 2)

def PCF_RATIO(price, cash_flow_per_share):
    """
    Calculate the Price-to-Cash Flow (P/CF) ratio.

    The P/CF ratio is calculated as the current price per share divided by 
    the cash flow per share.

    Parameters:
    price (float): The price per share of the stock.
    cash_flow_per_share (float): The cash flow per share of the stock.

    Returns:
    float: The P/CF ratio, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if price <= 0 or cash_flow_per_share <= 0:
        raise ValueError("All input values must be positive.")

    pcf_ratio = price / cash_flow_per_share
    return np.round(pcf_ratio, 2)

def PE_RATIO(price, earnings_per_share):
    """
    Calculate the Price-to-Earnings (P/E) ratio.

    The P/E ratio is calculated as the current price per share divided by 
    the earnings per share (EPS).

    Parameters:
    price (float): The price per share of the stock.
    earnings_per_share (float): The earnings per share of the stock.

    Returns:
    float: The P/E ratio, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if price <= 0 or earnings_per_share <= 0:
        raise ValueError("All input values must be positive.")

    pe_ratio = price / earnings_per_share
    return np.round(pe_ratio, 2)

def PEG_RATIO(pe_ratio, earnings_growth_rate):
    """
    Calculate the Price-to-Earnings-to-Growth (PEG) ratio.

    The PEG ratio is calculated as the P/E ratio divided by 
    the earnings growth rate.

    Parameters:
    pe_ratio (float): The Price-to-Earnings ratio.
    earnings_growth_rate (float): The annual earnings growth rate (as a percentage).

    Returns:
    float: The PEG ratio, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if pe_ratio <= 0 or earnings_growth_rate <= 0:
        raise ValueError("All input values must be positive.")

    peg_ratio = pe_ratio / earnings_growth_rate
    return np.round(peg_ratio, 2)

def TTM_PS_RATIO(price, total_revenue, shares_outstanding):
    """
    Calculate the Trailing Twelve Months Price-to-Sales (TTM P/S) ratio.

    The TTM P/S ratio is calculated as the current price per share divided by 
    the sales per share over the trailing twelve months.

    Parameters:
    price (float): The price per share of the stock.
    total_revenue (float): The trailing twelve months' total revenue (in the same currency as price).
    shares_outstanding (float): The number of shares outstanding.

    Returns:
    float: The TTM P/S ratio, rounded to two decimal places.

    Raises:
    ValueError: If any of the provided inputs are non-positive.
    """
    if price <= 0 or total_revenue <= 0 or shares_outstanding <= 0:
        raise ValueError("All input values must be positive.")

    # Calculate sales per share
    sales_per_share = total_revenue / shares_outstanding

    # Calculate TTM P/S ratio
    ps_ratio = price / sales_per_share

    return np.round(ps_ratio, 2)