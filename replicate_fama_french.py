import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from helper import factor_bucket, setup_logging, size_bucket, weighted_average


def compute_mkt_factor(input_crsp_filename: str,
                       output_mkt_factor_filename: str,
                       logging_enabled: bool = True):
    """
    Compute the Mkt factor.

    Parameters:
        input_crsp_filename (str): The file path to the CRSP data.
        output_mkt_factor_filename (str): The file path to save the Mkt factor.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """
    
    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the raw CRSP data, filtering for the relevant columns to save memory, and parsing the date columns...")
    crsp = pd.read_csv(filepath_or_buffer=input_crsp_filename,
                       usecols=['month_end_date', 'delisting_adjusted_monthly_return', 'weight'],
                       parse_dates=['month_end_date'])
    
    logging.info("Filtering the CRSP data to only include data where the weight is greater than zero and the delisting adjusted monthly return is not null...")
    universe = crsp[(crsp['weight'] > 0) &
                    (crsp['delisting_adjusted_monthly_return'].notnull())]
    
    logging.info("Calculating the value-weighted return...")
    vwret = universe.groupby(['month_end_date']).apply(weighted_average, 'delisting_adjusted_monthly_return', 'weight').reset_index().rename(columns={0: 'vwret'})

    logging.info("Renaming the columns to be more descriptive...")
    vwret = vwret.rename(columns={'month_end_date': 'date',
                                  'vwret': 'xMkt'})

    logging.info("Saving the Mkt factor to a CSV file...")
    vwret.to_csv(output_mkt_factor_filename, index=False)


def compute_hml_factor(input_ccm_filename: str,
                       input_crsp_filename: str,
                       output_hml_factor_filename: str,
                       logging_enabled: bool = True):
    """
    Compute the HML factor.

    Parameters:
        input_ccm_filename (str): The file path to the CCM June data.
        input_crsp_filename (str): The file path to the CRSP data.
        output_hml_factor_filename (str): The file path to save the HML factor.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """
    
    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the raw CCM June data, filtering for the relevant columns to save memory, and parsing the date columns...")
    crsp_compustat_june = pd.read_csv(filepath_or_buffer=input_ccm_filename,
                                      usecols=['month_end_date', 'permanent_number', 'BOOK_EQUITY', 'december_market_equity', 'exchange_code', 'market_equity'],
                                      parse_dates=['month_end_date'])
    
    logging.info("Reading in the raw CRSP data, filtering for the relevant columns to save memory, and parsing the date columns...")
    crsp = pd.read_csv(filepath_or_buffer=input_crsp_filename,
                       usecols=['month_end_date', 'permanent_number', 'exchange_code', 'delisting_adjusted_monthly_return', 'market_equity', 'weight', 'fama_french_year'],
                       parse_dates=['month_end_date'])    
        
    logging.info("Calculating the book equity to December market equity ratio...")
    crsp_compustat_june['book_equity_to_december_market_equity'] = crsp_compustat_june['BOOK_EQUITY'] * 1000 / crsp_compustat_june['december_market_equity']

    logging.info("Creating a universe of NYSE stocks with positive book equity to December market equity and market equity...")
    nyse_hml = crsp_compustat_june[(crsp_compustat_june['exchange_code'] == 1) &
                                   (crsp_compustat_june['book_equity_to_december_market_equity'] > 0) &
                                   (crsp_compustat_june['market_equity'] > 0)]

    logging.info("Creating the market equity breakpoints...")
    nyse_market_equity_breakpoints = nyse_hml.groupby(['month_end_date'])['market_equity'].median().reset_index().rename(columns={'market_equity': 'market_equity_median'})

    logging.info("Creating the book equity to December market equity breakpoints...")
    nyse_book_equity_to_market_equity_breakpoints = nyse_hml.groupby(['month_end_date'])['book_equity_to_december_market_equity'].describe(percentiles=[0.3, 0.7]).reset_index()

    logging.info("Keeping only the 30% and 70% percentiles for the book equity to December market equity breakpoints...")
    nyse_book_equity_to_market_equity_breakpoints = nyse_book_equity_to_market_equity_breakpoints[['month_end_date', '30%', '70%']]

    logging.info("Merging the market equity and book equity to market equity breakpoints...")
    nyse_breakpoints = pd.merge(nyse_market_equity_breakpoints,
                                nyse_book_equity_to_market_equity_breakpoints,
                                how='inner',
                                on=['month_end_date'])

    logging.info("Merging the CRSP Compustat June data with the breakpoints...")
    crsp_compustat_june = pd.merge(crsp_compustat_june,
                                   nyse_breakpoints,
                                   how='left',
                                   left_on='month_end_date',
                                   right_on='month_end_date')

    logging.info("Creating the size portfolios...")
    crsp_compustat_june['size_portfolio'] = np.where((crsp_compustat_june['book_equity_to_december_market_equity'] > 0) &
                                                     (crsp_compustat_june['market_equity'] > 0),
                                                     crsp_compustat_june.apply(size_bucket, axis=1),
                                                     '')

    logging.info("Creating the value portfolios...")
    crsp_compustat_june['factor_portfolio'] = np.where((crsp_compustat_june['book_equity_to_december_market_equity'] > 0) &
                                                       (crsp_compustat_june['market_equity'] > 0),
                                                       crsp_compustat_june.apply(lambda row: factor_bucket(row, 'book_equity_to_december_market_equity'), axis=1),
                                                       '')

    logging.info("Creating a column for the Fama-French year...")
    crsp_compustat_june['fama_french_year'] = crsp_compustat_june['month_end_date'].dt.year

    logging.info("Merging the CRSP data with the CRSP Compustat June data...")
    crsp_compustat = pd.merge(crsp,
                              crsp_compustat_june[['permanent_number', 'fama_french_year', 'size_portfolio', 'factor_portfolio']],
                              how='left',
                              on=['permanent_number', 'fama_french_year'])
    
    logging.info("Calculating the value weighted returns...")
    value_weighted_returns = crsp_compustat.groupby(['month_end_date', 'size_portfolio', 'factor_portfolio']).apply(weighted_average, 'delisting_adjusted_monthly_return', 'weight').reset_index().rename(columns={0: 'value_weighted_return'})

    logging.info("Creating the combined size value portfolios...")
    value_weighted_returns['size_factor_portfolio'] = value_weighted_returns['size_portfolio'] + value_weighted_returns['factor_portfolio']

    logging.info("Creating the Fama-French replicated factors...")
    fama_french_replicated_factors = value_weighted_returns.pivot(index='month_end_date', columns='size_factor_portfolio', values='value_weighted_return').reset_index()

    logging.info("Calculating the xHML factor...")
    fama_french_replicated_factors['H'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['SH']) / 2
    fama_french_replicated_factors['L'] = (fama_french_replicated_factors['BL'] + fama_french_replicated_factors['SL']) / 2
    fama_french_replicated_factors['xHML'] = fama_french_replicated_factors['H'] - fama_french_replicated_factors['L']

    logging.info("Calculating the xSHML factor..")
    fama_french_replicated_factors['S'] = (fama_french_replicated_factors['SH'] + fama_french_replicated_factors['SM'] + fama_french_replicated_factors['SL']) / 3
    fama_french_replicated_factors['B'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['BM'] + fama_french_replicated_factors['BL']) / 3
    fama_french_replicated_factors['xSHML'] = fama_french_replicated_factors['S'] - fama_french_replicated_factors['B']

    logging.info("Keeping only the date and the xHML factor and xSHML factor...")
    fama_french_replicated_factors = fama_french_replicated_factors[['month_end_date', 'xHML', 'xSHML']]

    logging.info("Renaming the month_end_date column to date...")
    fama_french_replicated_factors = fama_french_replicated_factors.rename(columns={'month_end_date': 'date'})

    logging.info("Saving the Fama-French replicated factors to a CSV file...")
    fama_french_replicated_factors.to_csv(output_hml_factor_filename, index=False)


def compute_rmw_factor(input_ccm_filename: str,
                       input_crsp_filename: str,
                       output_rmw_factor_filename: str,
                       logging_enabled: bool = True):
    """
    Compute the RMW factor.

    Parameters:
        input_ccm_filename (str): The file path to the CCM June data.
        input_crsp_filename (str): The file path to the CRSP data.
        output_rmw_factor_filename (str): The file path to save the RMW factor.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """

    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the raw CCM June data and parsing the date columns...")
    crsp_compustat_june = pd.read_csv(filepath_or_buffer=input_ccm_filename,
                                      usecols=['month_end_date', 'permanent_number', 'BOOK_EQUITY', 'OPERATING_PROFITABILITY', 'december_market_equity', 'exchange_code', 'market_equity', 'revenue', 'cost_of_goods_sold', 'interest_expense', 'selling_general_and_administrative_expenses'],
                                      parse_dates=['month_end_date'])

    logging.info("Reading in the raw CRSP data and parsing the date columns...")
    crsp = pd.read_csv(filepath_or_buffer=input_crsp_filename,
                       usecols=['month_end_date', 'permanent_number', 'exchange_code', 'delisting_adjusted_monthly_return', 'market_equity', 'weight', 'fama_french_year'],
                       parse_dates=['month_end_date'])    

    logging.info("Creating a universe of NYSE common stocks with positive book equity, positive market equity, positive December market equity, non-missing revenue, and non-missing one of cost of goods sold, interest expense, or selling general and administrative expenses...")
    nyse_rmw = crsp_compustat_june[(crsp_compustat_june['exchange_code'] == 1) &
                                   (crsp_compustat_june['BOOK_EQUITY'] > 0) &
                                   (crsp_compustat_june['market_equity'] > 0) &
                                   (crsp_compustat_june['december_market_equity'] > 0) &
                                   (crsp_compustat_june['revenue'].notnull()) &
                                   ((crsp_compustat_june['cost_of_goods_sold'].notnull()) |
                                    (crsp_compustat_june['interest_expense'].notnull()) |
                                    (crsp_compustat_june['selling_general_and_administrative_expenses'].notnull()))]

    logging.info("Creating the market equity breakpoints...")
    nyse_market_equity_breakpoints = nyse_rmw.groupby(['month_end_date'])['market_equity'].median().reset_index().rename(columns={'market_equity': 'market_equity_median'})

    logging.info("Creating the operating profitability breakpoints...")
    nyse_operating_profitability_breakpoints = nyse_rmw.groupby(['month_end_date'])['OPERATING_PROFITABILITY'].describe(percentiles=[0.3, 0.7]).reset_index()

    logging.info("Keeping only the 30% and 70% percentiles for the operating profitability breakpoints...")
    nyse_operating_profitability_breakpoints = nyse_operating_profitability_breakpoints[['month_end_date', '30%', '70%']]

    logging.info("Merging the market equity and operating profitability breakpoints...")
    nyse_breakpoints = pd.merge(nyse_market_equity_breakpoints,
                                nyse_operating_profitability_breakpoints,
                                how='inner',
                                on=['month_end_date'])

    logging.info("Merging the CRSP Compustat June data with the breakpoints...")
    crsp_compustat_june = pd.merge(crsp_compustat_june,
                                   nyse_breakpoints,
                                   how='left',
                                   left_on='month_end_date',
                                   right_on='month_end_date')

    logging.info("Creating the size portfolios...")
    crsp_compustat_june['size_portfolio'] = np.where((crsp_compustat_june['BOOK_EQUITY'] > 0) &
                                                     (crsp_compustat_june['market_equity'] > 0) &
                                                     (crsp_compustat_june['december_market_equity'] > 0) &
                                                     (crsp_compustat_june['revenue'].notnull()) &
                                                     ((crsp_compustat_june['cost_of_goods_sold'].notnull()) |
                                                      (crsp_compustat_june['interest_expense'].notnull()) |
                                                      (crsp_compustat_june['selling_general_and_administrative_expenses'].notnull())),
                                                      crsp_compustat_june.apply(size_bucket, axis=1),
                                                      '')

    logging.info("Creating the operating profitability portfolios...")
    crsp_compustat_june['factor_portfolio'] = np.where((crsp_compustat_june['BOOK_EQUITY'] > 0) &
                                                       (crsp_compustat_june['market_equity'] > 0) &
                                                       (crsp_compustat_june['december_market_equity'] > 0) &
                                                       (crsp_compustat_june['revenue'].notnull()) &
                                                       ((crsp_compustat_june['cost_of_goods_sold'].notnull()) |
                                                        (crsp_compustat_june['interest_expense'].notnull()) |
                                                        (crsp_compustat_june['selling_general_and_administrative_expenses'].notnull())),
                                                        crsp_compustat_june.apply(lambda row: factor_bucket(row, 'OPERATING_PROFITABILITY'), axis=1),
                                                        '')

    logging.info("Creating a column for the Fama-French year...")
    crsp_compustat_june['fama_french_year'] = crsp_compustat_june['month_end_date'].dt.year

    logging.info("Merging the CRSP data with the CRSP Compustat June data...")
    crsp_compustat = pd.merge(crsp,
                              crsp_compustat_june[['permanent_number', 'fama_french_year', 'size_portfolio', 'factor_portfolio']],
                              how='left',
                              on=['permanent_number', 'fama_french_year'])
    
    logging.info("Calculating the value weighted returns...")
    value_weighted_returns = crsp_compustat.groupby(['month_end_date', 'size_portfolio', 'factor_portfolio']).apply(weighted_average, 'delisting_adjusted_monthly_return', 'weight').reset_index().rename(columns={0: 'value_weighted_return'})

    logging.info("Creating the combined size operating profitability portfolios...")
    value_weighted_returns['size_factor_portfolio'] = value_weighted_returns['size_portfolio'] + value_weighted_returns['factor_portfolio']

    logging.info("Creating the Fama-French replicated factors...")
    fama_french_replicated_factors = value_weighted_returns.pivot(index='month_end_date', columns='size_factor_portfolio', values='value_weighted_return').reset_index()

    logging.info("Calculating the xRMW factor...")
    fama_french_replicated_factors['H'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['SH']) / 2
    fama_french_replicated_factors['L'] = (fama_french_replicated_factors['BL'] + fama_french_replicated_factors['SL']) / 2
    fama_french_replicated_factors['xRMW'] = fama_french_replicated_factors['H'] - fama_french_replicated_factors['L']

    logging.info("Calculating the xSRMW factor..")
    fama_french_replicated_factors['S'] = (fama_french_replicated_factors['SH'] + fama_french_replicated_factors['SM'] + fama_french_replicated_factors['SL']) / 3
    fama_french_replicated_factors['B'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['BM'] + fama_french_replicated_factors['BL']) / 3
    fama_french_replicated_factors['xSRMW'] = fama_french_replicated_factors['S'] - fama_french_replicated_factors['B']

    logging.info("Keeping only the date and the xRMW factor and xSRMW factor...")
    fama_french_replicated_factors = fama_french_replicated_factors[['month_end_date', 'xRMW', 'xSRMW']]

    logging.info("Renaming the month_end_date column to date...")
    fama_french_replicated_factors = fama_french_replicated_factors.rename(columns={'month_end_date': 'date'})

    logging.info("Saving the Fama-French replicated factors to a CSV file...")
    fama_french_replicated_factors.to_csv(output_rmw_factor_filename, index=False)


def compute_cma_factor(input_ccm_filename: str,
                       input_crsp_filename: str,
                       output_cma_factor_filename: str,
                       logging_enabled: bool = True):
    """
    Compute the CMA factor.

    Parameters:
        input_ccm_filename (str): The file path to the CCM June data.
        input_crsp_filename (str): The file path to the CRSP data.
        output_cma_factor_filename (str): The file path to save the CMA factor.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """
    
    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the raw CCM June data and parsing the date columns...")
    crsp_compustat_june = pd.read_csv(filepath_or_buffer=input_ccm_filename,
                                      usecols=['month_end_date', 'permanent_number', 'INVESTMENT', 'december_market_equity', 'exchange_code', 'market_equity'],
                                      parse_dates=['month_end_date'])
    
    logging.info("Reading in the raw CRSP data and parsing the date columns...")
    crsp = pd.read_csv(filepath_or_buffer=input_crsp_filename,
                       usecols=['month_end_date', 'permanent_number', 'exchange_code', 'delisting_adjusted_monthly_return', 'market_equity', 'weight', 'fama_french_year'],
                       parse_dates=['month_end_date'])    
        
    logging.info("Creating a universe of NYSE common stocks with positive investment, positive market equity, and positive December market equity...")
    nyse_cma = crsp_compustat_june[(crsp_compustat_june['exchange_code'] == 1) &
                                   (crsp_compustat_june['december_market_equity'] > 0) &
                                   (crsp_compustat_june['market_equity'] > 0) & 
                                   (crsp_compustat_june['INVESTMENT'].notnull())]

    logging.info("Creating the market equity breakpoints...")
    nyse_market_equity_breakpoints = nyse_cma.groupby(['month_end_date'])['market_equity'].median().reset_index().rename(columns={'market_equity': 'market_equity_median'})

    logging.info("Creating the investment breakpoints...")
    nyse_investment_breakpoints = nyse_cma.groupby(['month_end_date'])['INVESTMENT'].describe(percentiles=[0.3, 0.7]).reset_index()

    logging.info("Keeping only the 30% and 70% percentiles for the investment breakpoints...")
    nyse_investment_breakpoints = nyse_investment_breakpoints[['month_end_date', '30%', '70%']]

    logging.info("Merging the market equity and investment breakpoints...")
    nyse_breakpoints = pd.merge(nyse_market_equity_breakpoints,
                                nyse_investment_breakpoints,
                                how='inner',
                                on=['month_end_date'])

    logging.info("Merging the CRSP Compustat June data with the breakpoints...")
    crsp_compustat_june = pd.merge(crsp_compustat_june,
                                   nyse_breakpoints,
                                   how='left',
                                   left_on='month_end_date',
                                   right_on='month_end_date')

    logging.info("Creating the size portfolios...")
    crsp_compustat_june['size_portfolio'] = np.where((crsp_compustat_june['market_equity'] > 0) &
                                                     (crsp_compustat_june['december_market_equity'] > 0),
                                                     crsp_compustat_june.apply(size_bucket, axis=1),
                                                     '')

    logging.info("Creating the investment portfolios...")
    crsp_compustat_june['factor_portfolio'] = np.where((crsp_compustat_june['market_equity'] > 0) &
                                                       (crsp_compustat_june['december_market_equity'] > 0),
                                                       crsp_compustat_june.apply(lambda row: factor_bucket(row, 'INVESTMENT'), axis=1),
                                                       '')

    logging.info("Creating a column for the Fama-French year...")
    crsp_compustat_june['fama_french_year'] = crsp_compustat_june['month_end_date'].dt.year

    logging.info("Merging the CRSP data with the CRSP Compustat June data...")
    crsp_compustat = pd.merge(crsp,
                              crsp_compustat_june[['permanent_number', 'fama_french_year', 'size_portfolio', 'factor_portfolio']],
                              how='left',
                              on=['permanent_number', 'fama_french_year'])

    logging.info("Calculating the value weighted returns...")
    value_weighted_returns = crsp_compustat.groupby(['month_end_date', 'size_portfolio', 'factor_portfolio']).apply(weighted_average, 'delisting_adjusted_monthly_return', 'weight').reset_index().rename(columns={0: 'value_weighted_return'})

    logging.info("Creating the combined size investment portfolios...")
    value_weighted_returns['size_factor_portfolio'] = value_weighted_returns['size_portfolio'] + value_weighted_returns['factor_portfolio']

    logging.info("Creating the Fama-French replicated factors...")
    fama_french_replicated_factors = value_weighted_returns.pivot(index='month_end_date', columns='size_factor_portfolio', values='value_weighted_return').reset_index()

    logging.info("Calculating the xCMA factor...")
    fama_french_replicated_factors['A'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['SH']) / 2
    fama_french_replicated_factors['C'] = (fama_french_replicated_factors['BL'] + fama_french_replicated_factors['SL']) / 2
    fama_french_replicated_factors['xCMA'] = fama_french_replicated_factors['C'] - fama_french_replicated_factors['A']

    logging.info("Calculating the xSCMA factor..")
    fama_french_replicated_factors['S'] = (fama_french_replicated_factors['SH'] + fama_french_replicated_factors['SM'] + fama_french_replicated_factors['SL']) / 3
    fama_french_replicated_factors['B'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['BM'] + fama_french_replicated_factors['BL']) / 3
    fama_french_replicated_factors['xSCMA'] = fama_french_replicated_factors['S'] - fama_french_replicated_factors['B']

    logging.info("Keeping only the date and the xCMA factor and xSCMA factor...")
    fama_french_replicated_factors = fama_french_replicated_factors[['month_end_date', 'xCMA', 'xSCMA']]

    logging.info("Renaming the month_end_date column to date...")
    fama_french_replicated_factors = fama_french_replicated_factors.rename(columns={'month_end_date': 'date'})

    logging.info("Saving the Fama-French replicated factors to a CSV file...")
    fama_french_replicated_factors.to_csv(output_cma_factor_filename, index=False)


def compute_umd_factor(input_crsp_filename: str,
                       output_umd_factor_filename: str,
                       logging_enabled: bool = True):
    """
    Compute the UMD factor.

    Parameters:
        input_crsp_filename (str): The file path to the CRSP data.
        output_umd_factor_filename (str): The file path to save the UMD factor.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """
    
    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the raw CRSP data, filtering for the relevant columns to save memory, and parsing the date columns...")
    crsp = pd.read_csv(filepath_or_buffer=input_crsp_filename,
                       usecols=['permanent_number', 'month_end_date', 'delisting_adjusted_monthly_return', 'weight', 'market_equity', 'exchange_code', 'monthly_price'],
                          parse_dates=['month_end_date'])
    
    logging.info("Calculating momentum...")
    crsp['MOMENTUM'] = crsp.groupby('permanent_number')['delisting_adjusted_monthly_return'].apply(
        lambda x: (1 + x.shift(2)).rolling(window=11, min_periods=11).apply(np.prod) - 1
    ).reset_index(0, drop=True)

    logging.info("Creating a universe of NYSE common stocks with positive market equity, non-missing monthly price, non-missing delisting adjusted monthly returns, and non-missing momentum...")
    nyse_umd = crsp[(crsp['weight'] > 0) &
                    (crsp['monthly_price'].shift(12).notnull()) &
                    (crsp['delisting_adjusted_monthly_return'].shift(1).notnull()) &
                    (crsp['exchange_code'] == 1) &
                    (crsp['MOMENTUM'].notnull())]
    
    logging.info("Calculating the market equity breakpoints...")
    nyse_market_equity_breakpoints = nyse_umd.groupby(['month_end_date'])['market_equity'].median().reset_index().rename(columns={'market_equity': 'market_equity_median'})

    logging.info("Calculating the momentum breakpoints...")
    nyse_momentum_breakpoints = nyse_umd.groupby(['month_end_date'])['MOMENTUM'].describe(percentiles=[0.3, 0.7]).reset_index()

    logging.info("Keeping only the 30% and 70% percentiles for the momentum breakpoints...")
    nyse_momentum_breakpoints = nyse_momentum_breakpoints[['month_end_date', '30%', '70%']]

    logging.info("Merging the market equity and momentum breakpoints...")
    nyse_breakpoints = pd.merge(nyse_market_equity_breakpoints,
                                nyse_momentum_breakpoints,
                                how='inner',
                                on=['month_end_date'])
    
    logging.info("Merging the CRSP data with the breakpoints...")
    crsp = pd.merge(crsp,
                    nyse_breakpoints,
                    how='left',
                    on='month_end_date')
    
    logging.info("Creating the size portfolios...")
    crsp['size_portfolio'] = np.where(crsp['market_equity'] > 0 &
                                     (crsp['monthly_price'].shift(12).notnull()) &
                                     (crsp['delisting_adjusted_monthly_return'].shift(1).notnull()),
                                     crsp.apply(size_bucket, axis=1),
                                     '')
    
    logging.info("Creating the momentum portfolios...")
    crsp['factor_portfolio'] = np.where(crsp['MOMENTUM'].notnull() &
                                        (crsp['monthly_price'].shift(12).notnull()) &
                                        (crsp['delisting_adjusted_monthly_return'].shift(1).notnull()),
                                        crsp.apply(lambda row: factor_bucket(row, 'MOMENTUM'), axis=1),
                                        '')
    
    logging.info("Creating a column for the Fama-French year...")
    crsp['fama_french_year'] = crsp['month_end_date'].dt.year
    
    logging.info("Calculating the size and momentum weighted returns...")
    value_weighted_returns = crsp.groupby(['month_end_date', 'size_portfolio', 'factor_portfolio']).apply(weighted_average, 'delisting_adjusted_monthly_return', 'weight').reset_index().rename(columns={0: 'value_weighted_return'})

    logging.info("Creating the combined size momentum portfolios...")
    value_weighted_returns['size_factor_portfolio'] = value_weighted_returns['size_portfolio'] + value_weighted_returns['factor_portfolio']

    logging.info("Creating the Fama-French replicated factors...")
    fama_french_replicated_factors = value_weighted_returns.pivot(index='month_end_date', columns='size_factor_portfolio', values='value_weighted_return').reset_index()

    logging.info("Calculating the xUMD factor...")
    fama_french_replicated_factors['U'] = (fama_french_replicated_factors['BH'] + fama_french_replicated_factors['SH']) / 2
    fama_french_replicated_factors['D'] = (fama_french_replicated_factors['BL'] + fama_french_replicated_factors['SL']) / 2
    fama_french_replicated_factors['xUMD'] = fama_french_replicated_factors['U'] - fama_french_replicated_factors['D']

    logging.info("Keeping only the date and the xUMD factor...")
    fama_french_replicated_factors = fama_french_replicated_factors[['month_end_date', 'xUMD']]

    logging.info("Renaming the month_end_date column to date...")
    fama_french_replicated_factors = fama_french_replicated_factors.rename(columns={'month_end_date': 'date'})

    logging.info("Saving the Fama-French replicated factors to a CSV file...")
    fama_french_replicated_factors.to_csv(output_umd_factor_filename, index=False)


def compare_fama_french_factors(input_fama_french_3_factors_filename: str,
                                input_mkt_factor_filename: str,
                                input_hml_factor_filename: str,
                                input_rmw_factor_filename: str,
                                input_cma_factor_filename: str,
                                input_umd_factor_filename: str,
                                logging_enabled: bool = True):
    """
    Compare the Fama-French factors with the replicated factors.

    Parameters:
        input_fama_french_3_factors_filename (str): The file path to the Fama-French 3 factors.
        input_mkt_factor_filename (str): The file path to the Mkt factor.
        input_hml_factor_filename (str): The file path to the HML factor.
        input_rmw_factor_filename (str): The file path to the RMW factor.
        input_cma_factor_filename (str): The file path to the CMA factor.
        input_umd_factor_filename (str): The file path to the UMD factor.

    Returns:
        None
    """
    
    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the Fama-French factors, filtering for the relevant columns to save memory, and parsing the date column...")
    fama_french_three_factors = pd.read_csv(filepath_or_buffer=input_fama_french_3_factors_filename,
                                            parse_dates=['date'])

    logging.info("Reading in the Mkt factor and parsing the date column...")
    mkt_factor = pd.read_csv(filepath_or_buffer=input_mkt_factor_filename,
                             parse_dates=['date'])
    
    logging.info("Reading in the HML factor and parsing the date column...")
    hml_factor = pd.read_csv(filepath_or_buffer=input_hml_factor_filename,
                                parse_dates=['date'])
    
    logging.info("Reading in the RMW factor and parsing the date column...")
    rmw_factor = pd.read_csv(filepath_or_buffer=input_rmw_factor_filename,
                                parse_dates=['date'])
    
    logging.info("Reading in the CMA factor and parsing the date column...")
    cma_factor = pd.read_csv(filepath_or_buffer=input_cma_factor_filename,
                                parse_dates=['date'])
    
    logging.info("Reading in the UMD factor and parsing the date column...")
    umd_factor = pd.read_csv(filepath_or_buffer=input_umd_factor_filename,
                                parse_dates=['date'])

    logging.info("Parsing the date column for the Fama-French factors...")
    fama_french_three_factors['date'] = pd.to_datetime(fama_french_three_factors['date'], format='%Y%m') + pd.offsets.MonthEnd(0)
    
    logging.info("Merging the Fama-French factors with the replicated Mkt factor...")
    fama_french_comparision = pd.merge(fama_french_three_factors,
                                       mkt_factor,
                                       how='inner',
                                       on='date')
    
    logging.info("Merging the Fama-French factors with the replicated HML factor...")
    fama_french_comparision = pd.merge(fama_french_comparision,
                                       hml_factor,
                                       how='inner',
                                       on='date')
    
    logging.info("Merging the Fama-French factors with the replicated RMW factor...")
    fama_french_comparision = pd.merge(fama_french_comparision,
                                       rmw_factor,
                                       how='inner',
                                       on='date')
    
    logging.info("Merging the Fama-French factors with the replicated CMA factor...")
    fama_french_comparision = pd.merge(fama_french_comparision,
                                       cma_factor,
                                       how='inner',
                                       on='date')
    
    logging.info("Merging the Fama-French factors with the replicated UMD factor...")
    fama_french_comparision = pd.merge(fama_french_comparision,
                                       umd_factor,
                                       how='inner',
                                       on='date')

    logging.info("Filtering the Fama-French factors to only include data from 1963 onwards...")
    fama_french_comparision = fama_french_comparision[(fama_french_comparision['date'] >= '01/01/1963') & (fama_french_comparision['date'] <= '12/31/2023')]

    logging.info("Subtracting the risk-free rate from the Mkt factor...")
    fama_french_comparision['xMkt-Rf'] = ((fama_french_comparision['xMkt'] - fama_french_comparision['RF'] / 100) * 100).round(2)

    logging.info("Calculating the SMB factor...")
    fama_french_comparision['xSMB'] = ((fama_french_comparision['xSHML'] + fama_french_comparision['xSRMW'] + fama_french_comparision['xSCMA']) / 3 * 100).round(2) 

    logging.info("Dividing the replicated factors by 100 and rounding to 2 decimal places...")
    fama_french_comparision['xHML'] = (fama_french_comparision['xHML'] * 100).round(2)
    fama_french_comparision['xRMW'] = (fama_french_comparision['xRMW'] * 100).round(2)
    fama_french_comparision['xCMA'] = (fama_french_comparision['xCMA'] * 100).round(2)
    fama_french_comparision['xUMD'] = (fama_french_comparision['xUMD'] * 100).round(2)

    print(f"Mkt-Rf | Pearson coreelation={stats.pearsonr(fama_french_comparision['Mkt-RF'], fama_french_comparision['xMkt-Rf'])[0]:.5f} | Spearman correlation={stats.spearmanr(fama_french_comparision['Mkt-RF'], fama_french_comparision['xMkt-Rf'])[0]:.5f} | MAE={np.mean(np.abs(fama_french_comparision['Mkt-RF'] - fama_french_comparision['xMkt-Rf'])):.5f}")
    print(f"SMB | Pearson coreelation={stats.pearsonr(fama_french_comparision['SMB'], fama_french_comparision['xSMB'])[0]:.5f} | Spearman correlation={stats.spearmanr(fama_french_comparision['SMB'], fama_french_comparision['xSMB'])[0]:.5f} | MAE={np.mean(np.abs(fama_french_comparision['SMB'] - fama_french_comparision['xSMB'])):.5f}")
    print(f"HML | Pearson coreelation={stats.pearsonr(fama_french_comparision['HML'], fama_french_comparision['xHML'])[0]:.5f} | Spearman correlation={stats.spearmanr(fama_french_comparision['HML'], fama_french_comparision['xHML'])[0]:.5f} | MAE={np.mean(np.abs(fama_french_comparision['HML'] - fama_french_comparision['xHML'])):.5f}")
    print(f"RMW | Pearson coreelation={stats.pearsonr(fama_french_comparision['RMW'], fama_french_comparision['xRMW'])[0]:.5f} | Spearman correlation={stats.spearmanr(fama_french_comparision['RMW'], fama_french_comparision['xRMW'])[0]:.5f} | MAE={np.mean(np.abs(fama_french_comparision['RMW'] - fama_french_comparision['xRMW'])):.5f}")
    print(f"CMA | Pearson coreelation={stats.pearsonr(fama_french_comparision['CMA'], fama_french_comparision['xCMA'])[0]:.5f} | Spearman correlation={stats.spearmanr(fama_french_comparision['CMA'], fama_french_comparision['xCMA'])[0]:.5f} | MAE={np.mean(np.abs(fama_french_comparision['CMA'] - fama_french_comparision['xCMA'])):.5f}")
    print(f"UMD | Pearson coreelation={stats.pearsonr(fama_french_comparision['UMD'], fama_french_comparision['xUMD'])[0]:.5f} | Spearman correlation={stats.spearmanr(fama_french_comparision['UMD'], fama_french_comparision['xUMD'])[0]:.5f} | MAE={np.mean(np.abs(fama_french_comparision['UMD'] - fama_french_comparision['xUMD'])):.5f}")


def replicate_fama_french():
    """
    Replicate the Fama-French factors using the CRSP and Compustat data.

    Parameters:
        None

    Returns:
        None
    """

    compute_mkt_factor(input_crsp_filename='data/processed_data/crsp.csv',
                       output_mkt_factor_filename='data/processed_data/mkt_factor.csv',
                       logging_enabled=True)
    
    compute_hml_factor(input_ccm_filename='data/processed_data/ccm.csv',
                       input_crsp_filename='data/processed_data/crsp.csv',
                       output_hml_factor_filename='data/processed_data/hml_factor.csv',
                       logging_enabled=True)
    
    compute_rmw_factor(input_ccm_filename='data/processed_data/ccm.csv',
                       input_crsp_filename='data/processed_data/crsp.csv',
                       output_rmw_factor_filename='data/processed_data/rmw_factor.csv',
                       logging_enabled=True)
    
    compute_cma_factor(input_ccm_filename='data/processed_data/ccm.csv',
                       input_crsp_filename='data/processed_data/crsp.csv',
                       output_cma_factor_filename='data/processed_data/cma_factor.csv',
                       logging_enabled=True)

    compute_umd_factor(input_crsp_filename='data/processed_data/crsp.csv',
                       output_umd_factor_filename='data/processed_data/umd_factor.csv',
                       logging_enabled=True)
    
    compare_fama_french_factors(input_fama_french_3_factors_filename='data/raw_data/raw_fama_french_6_factors.csv',
                                input_mkt_factor_filename='data/processed_data/mkt_factor.csv',
                                input_hml_factor_filename='data/processed_data/hml_factor.csv',
                                input_rmw_factor_filename='data/processed_data/rmw_factor.csv',
                                input_cma_factor_filename='data/processed_data/cma_factor.csv',
                                input_umd_factor_filename='data/processed_data/umd_factor.csv',
                                logging_enabled=True)
