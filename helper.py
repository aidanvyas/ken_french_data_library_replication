import logging
from functools import reduce
import pandas as pd
import numpy as np


def setup_logging(logging_enabled):
    """
    Helper function to setup logging.

    Parameters:
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """
    if logging_enabled:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    else:
        logging.disable(logging.CRITICAL)


def coalesce(*args):
    """
    Helper function that the first non-None value across given pandas Series for each row.

    Parameters:
        *args (pd.Series): A pandas series.

    Returns:
        value (pd.Series): A pandas series.
    """
    args = [arg if isinstance(arg, pd.Series) else pd.Series(arg) for arg in args]
    return reduce(lambda x, y: x.where(x.notnull(), y), args)


def size_bucket(row):
    """
    Helper function to assign a stock to the correct size bucket.

    Parameters:
        row (pd.Series): A pandas series.

    Returns:
        value (str): A string indicating the size bucket.
    """
    if row['market_equity'] == np.nan:
        value=''
    elif row['market_equity'] <= row['market_equity_median']:
        value='S'
    else:
        value='B'
    return value


def factor_bucket(row, factor):
    """
    Helper function to assign a stock to the correct factor bucket.

    Parameters:
        row (pd.Series): A pandas series.
        factor (str): A string indicating the factor.

    Returns:
        value (str): A string indicating the factor bucket.
    """
    if row[factor] <= row['30%']:
        value = 'L'
    elif row[factor] <= row['70%']:
        value = 'M'
    elif row[factor] > row['70%']:
        value = 'H'
    else:
        value = ''
    return value


def weighted_average(group, avg_name, weight_name):
    """
    Helper function to calculate the weighted average.

    Parameters:
        group (pd.DataFrame): A pandas dataframe.
        avg_name (str): A string indicating the average name.
        weight_name (str): A string indicating the weight name.

    Returns:
        value (float): A float indicating the weighted average.
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


def create_fama_french_portfolios(input_ccm_dataframe: pd.DataFrame,
                                  input_factor_name: str,
                                  save_size_portfolio: bool,
                                  is_inverse: bool = False) -> pd.DataFrame:
    """
    Helper function to create the Fama-French portfolios.

    Parameters:
        input_ccm_dataframe (pd.DataFrame): A pandas dataframe containing the monthly CRSP-Compustat merged data and portfolio assignments.
        input_factor_name (str): A string indicating the factor name.

    Returns:
        output_dataframe (pd.DataFrame): A pandas dataframe containing the Fama-French portfolios.
    """
    value_weighted_returns = input_ccm_dataframe.groupby(['month_end_date', 'size_portfolio', 'factor_portfolio']).apply(weighted_average, 'delisting_adjusted_monthly_return', 'weight').reset_index().rename(columns={0: 'value_weighted_return'})
    value_weighted_returns['size_factor_portfolio'] = value_weighted_returns['size_portfolio'] + value_weighted_returns['factor_portfolio']
    fama_french_replicated_factors = value_weighted_returns.pivot(index='month_end_date', columns='size_factor_portfolio', values='value_weighted_return').reset_index()
    fama_french_replicated_factors['H'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['SH']) / 2
    fama_french_replicated_factors['L'] = (fama_french_replicated_factors['BL'] + fama_french_replicated_factors['SL']) / 2
    if is_inverse:
        fama_french_replicated_factors['xHML'] = fama_french_replicated_factors['L'] - fama_french_replicated_factors['H']
    else:
        fama_french_replicated_factors['xHML'] = fama_french_replicated_factors['H'] - fama_french_replicated_factors['L']
    if save_size_portfolio:
        fama_french_replicated_factors['S'] = (fama_french_replicated_factors['SH'] + fama_french_replicated_factors['SM'] + fama_french_replicated_factors['SL']) / 3
        fama_french_replicated_factors['B'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['BM'] + fama_french_replicated_factors['BL']) / 3
        fama_french_replicated_factors['xSHML'] = fama_french_replicated_factors['S'] - fama_french_replicated_factors['B']
        fama_french_replicated_factors = fama_french_replicated_factors[['month_end_date', 'xHML', 'xSHML']]
        fama_french_replicated_factors = fama_french_replicated_factors.rename(columns={'month_end_date': 'date', 'xHML': 'x' + input_factor_name, 'xSHML': 'xS' + input_factor_name})
    else:
        fama_french_replicated_factors = fama_french_replicated_factors[['month_end_date', 'xHML']]
        fama_french_replicated_factors = fama_french_replicated_factors.rename(columns={'month_end_date': 'date', 'xHML': 'x' + input_factor_name})
    return fama_french_replicated_factors
