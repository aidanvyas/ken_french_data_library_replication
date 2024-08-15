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
