import logging
import numpy as np
import pandas as pd
from typing import List
from helper import setup_logging, coalesce


def process_compustat_data(input_fundamentals_annual_filename: str,
                           output_compustat_filename: str,
                           additional_columns: List[str],
                           logging_enabled: bool = True):
    """
    Process the Compustat data.

    Parameters:
        input_fundamentals_annual_filename (str): The file path to the raw Compustat fundamentals annual data.
        output_compustat_filename (str): The file path to save the processed Compustat data.
        additional_columns (List[str]): A list of additional columns to include in the processed Compustat data.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """
    
    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the raw Compustat data, filtering for the relevant columns to save memory, and parsing the date column...")
    compustat = pd.read_csv(filepath_or_buffer=input_fundamentals_annual_filename,
                            usecols=['gvkey', 'datadate', 'seq', 'ceq', 'txditc', 'pstkrv', 'pstkl', 'pstk', 'at', 'lt', 'revt', 'cogs', 'xint', 'xsga', 'mib', 'fyear', 'indfmt', 'datafmt', 'popsrc', 'consol'] + additional_columns,
                            parse_dates=['datadate'])
    
    logging.info("Renaming the columns to be more descriptive...")
    compustat = compustat.rename(columns={'gvkey': 'global_company_key',
                                          'datadate': 'month_end_date',
                                          'seq': 'shareholders_equity',
                                          'ceq': 'common/ordinary_equity',
                                          'txditc': 'deferred_taxes_and_investment_tax_credit',
                                          'pstkrv': 'preferred_stock_redemption_value',
                                          'pstkl': 'preferred_stock_liquidation_value',
                                          'pstk': 'total_preferred_stock',
                                          'at': 'total_assets',
                                          'lt': 'total_liabilities',
                                          'revt': 'revenue',
                                          'cogs': 'cost_of_goods_sold',
                                          'xint': 'interest_expense',
                                          'xsga': 'selling_general_and_administrative_expenses',
                                          'mib': 'minority_interest_balance_sheet',
                                          'fyear': 'fiscal_year',
                                          'indfmt': 'industry_format',
                                          'datafmt': 'data_format',
                                          'popsrc': 'population_source',
                                          'consol': 'consolidation_level'})

    # This should be done when you are downloading the data, but it is good to double check.
    # Specifically, these filters ensure that:
    # 1. the data is for industrial companies,
    # 2. the data is standardized,
    # 3. the population source is domestic,
    # 4. the data is consolidated, and 
    # 5. the data is from 1959 onwards.
    logging.info("Filtering the Compustat data to only include the relevant data...")
    compustat = compustat[(compustat['industry_format'] == 'INDL') &
                          (compustat['data_format'] == 'STD') &
                          (compustat['population_source'] == 'D') &
                          (compustat['consolidation_level'] == 'C') &
                          (compustat['month_end_date'] >= '01/01/1959')]

    # When I modify Compustat variables, I will capitalize the variable name to contrast it with the original variable name.
    logging.info("Creating a column for preferred stock, which equals the first non-null value of the preferred stock redemption value, preferred stock liquidation value, or total preferred stock...")
    compustat['PREFERRED_STOCK'] = coalesce(compustat['preferred_stock_redemption_value'],
                                            compustat['preferred_stock_liquidation_value'],
                                            compustat['total_preferred_stock'])
    
    logging.info("Creating a column for shareholders equity, which equals the first non-null value of shareholders equity, common/ordinary equity plus total preferred stock, or total assets minus total liabilities...")
    compustat['SHAREHOLDERS_EQUITY'] = coalesce(compustat['shareholders_equity'],
                                                (compustat['common/ordinary_equity'] + compustat['total_preferred_stock']),
                                                (compustat['total_assets'] - compustat['total_liabilities']))

    logging.info("Creating a column for book equity, which equals the sum of shareholders equity, deferred taxes and investment tax credit, minus preferred stock...")
    compustat['BOOK_EQUITY'] = compustat['SHAREHOLDERS_EQUITY'] + np.where(compustat['fiscal_year'] < 1993,
                                                                           compustat['deferred_taxes_and_investment_tax_credit'].fillna(0),
                                                                           0) - compustat['PREFERRED_STOCK'].fillna(0)
    
    logging.info("Creating a column for operating profitability, which equals revenue minus cost of goods sold, interest expense, and selling, general, and administrative expenses, divided by book equity plus minority interest from the balance sheet...")
    compustat['OPERATING_PROFITABILITY'] = (compustat['revenue'] - compustat['cost_of_goods_sold'].fillna(0) - compustat['interest_expense'].fillna(0) - compustat['selling_general_and_administrative_expenses'].fillna(0)) / (compustat['BOOK_EQUITY'] + compustat['minority_interest_balance_sheet'].fillna(0))

    logging.info("Creating a column for lagged total assets, which equals the total assets from the previous year...")
    compustat['LAGGED_TOTAL_ASSETS'] = compustat.sort_values(by=['global_company_key', 'month_end_date']).groupby(['global_company_key'])['total_assets'].shift(1)

    logging.info("Creating a column for investment, which equals the change in total assets divided by the lagged total assets...")
    compustat['INVESTMENT'] = (compustat['total_assets'] - compustat['LAGGED_TOTAL_ASSETS']) / compustat['LAGGED_TOTAL_ASSETS']

    logging.info("Filtering the Compustat data to only include the relevant columns...")
    compustat = compustat[['global_company_key', 'month_end_date', 'BOOK_EQUITY', 'OPERATING_PROFITABILITY', 'revenue', 'cost_of_goods_sold', 'interest_expense', 'selling_general_and_administrative_expenses', 'INVESTMENT']]

    logging.info("Saving the Compustat data to a CSV file...")
    compustat.to_csv(path_or_buf=output_compustat_filename, index=False)


def handle_delistings(input_delisting_information_filename: str,
                      crsp_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to handle delistings in the CRSP monthly stock file data.

    Parameters:
        input_delisting_information_filename (str): The file path to the raw CRSP delisting information data.
        crsp_monthly (pd.DataFrame): The CRSP monthly stock file data.

    Returns:
        crsp (pd.DataFrame): The CRSP monthly stock file data with delistings handled.
    """

    logging.info("Reading in the CRSP delisting information data, filtering for the relevant columns to save memory, and parsing the date column...")
    delisting_information = pd.read_csv(filepath_or_buffer=input_delisting_information_filename,
                                        usecols=['PERMNO', 'DLRET', 'DLSTDT'],
                                        parse_dates=['DLSTDT'])
    
    logging.info("Renaming the columns to be more descriptive...")
    delisting_information = delisting_information.rename(columns={'PERMNO': 'permanent_number',
                                                                  'DLRET': 'delisting_return',
                                                                  'DLSTDT': 'delisting_date'})

    logging.info("Calculating the month-end date for the delisting information...")
    delisting_information['month_end_date'] = pd.to_datetime(delisting_information['delisting_date']) + pd.offsets.MonthEnd(0)

    logging.info("Merging the CRSP monthly stock file and delisting information dataframes...")
    crsp = pd.merge(crsp_monthly,
                    delisting_information[['permanent_number', 'month_end_date', 'delisting_return']],
                    how='left',
                    on=['permanent_number', 'month_end_date'])

    logging.info("Setting the delisting and monthly returns to be zero if it is null...")
    crsp['delisting_return'] = crsp['delisting_return'].fillna(0)
    crsp['monthly_return'] = crsp['monthly_return'].fillna(0)

    logging.info("Coercing the delisting returns to be numeric...")
    crsp['delisting_return'] = pd.to_numeric(crsp['delisting_return'], errors='coerce')

    logging.info("Calculating the delisting adjusted monthly returns...")
    crsp['delisting_adjusted_monthly_return'] = (1 + crsp['monthly_return']) * (1 + crsp['delisting_return']) - 1

    logging.info("Dropping the monthly and delisting return columns...")
    crsp = crsp.drop(['monthly_return', 'delisting_return'], axis=1)

    logging.info("Returning the delisting adjusted CRSP monthly data...")
    return crsp


def handle_permanent_company_number(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to handle the permanent company number in the CRSP data.

    Parameters:
        crsp (pd.DataFrame): The CRSP data.

    Returns:
        crsp (pd.DataFrame): The CRSP data with the permanent company number handled.
    """

    # Essentially, we sum the market equity for each company, and then assign it to the security with the maximum market equity.
    logging.info("Creating a dataframe with the summed market equity for each company...")
    crsp_summed_market_equity = crsp.groupby(['month_end_date', 'permanent_company_number'])['market_equity'].sum().reset_index()

    logging.info("Creating a dataframe with the maximum market equity for each company...")
    crsp_maximum_market_equity = crsp.groupby(['month_end_date', 'permanent_company_number'])['market_equity'].max().reset_index()

    logging.info("Merging the maximum market equity dataframe with the CRSP dataframe to only keep the primary security...")
    crsp = pd.merge(crsp,
                    crsp_maximum_market_equity,
                    how='inner',
                    on=['month_end_date', 'permanent_company_number', 'market_equity'])

    logging.info("Dropping the market equity column...")
    crsp = crsp.drop(['market_equity'], axis=1)

    logging.info("Merging the summed market equity dataframe with the CRSP dataframe to assign the summed market equity to the primary security...")
    crsp = pd.merge(crsp,
                    crsp_summed_market_equity,
                    how='inner',
                    on=['month_end_date', 'permanent_company_number'])

    logging.info("Sorting the CRSP data by company and then date, dropping any duplicates...")
    crsp = crsp.sort_values(by=['permanent_company_number', 'month_end_date']).drop_duplicates()

    logging.info("Returning the CRSP data...")
    return crsp


def process_crsp_data(input_monthly_stock_file_filename: str,
                      input_historical_descriptive_information_filename: str,
                      input_delisting_information_filename: str,
                      output_crsp_filename: str,
                      output_crsp_june_filename: str,
                      logging_enabled: bool = True):
    """
    Process the CRSP data.

    Parameters:
        input_monthly_stock_file_filename (str): The file path to the raw CRSP monthly stock file data.
        input_historical_descriptive_information_filename (str): The file path to the raw CRSP historical descriptive information data.
        input_delisting_information_filename (str): The file path to the raw CRSP delisting information data.
        output_crsp_filename (str): The file path to save the processed CRSP data.
        output_crsp_june_filename (str): The file path to save the processed CRSP June data.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """

    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the raw CRSP monthly stock file data, filtering for the relevant columns to save memory, and parsing the date column...")
    monthly_stock_file = pd.read_csv(filepath_or_buffer=input_monthly_stock_file_filename,
                                     usecols=['PERMNO', 'PERMCO', 'MthCalDt', 'MthRet', 'MthRetx', 'MthPrc', 'ShrOut', 'USIncFlg'],
                                     parse_dates=['MthCalDt'])

    logging.info("Renaming the columns to be more descriptive...")
    monthly_stock_file = (monthly_stock_file.rename(columns={'PERMNO': 'permanent_number',
                                                             'PERMCO': 'permanent_company_number',
                                                             'MthCalDt': 'month_end_date',
                                                             'MthRet': 'monthly_return',
                                                             'MthPrc': 'monthly_price',
                                                             'ShrOut': 'shares_outstanding',
                                                             'USIncFlg': 'US_incorporation_flag'}))

    logging.info("Reading in the CRSP historical descriptive information data, filtering for the relevant columns to save memory, and parsing the date columns...")
    historical_descriptive_information = pd.read_csv(filepath_or_buffer=input_historical_descriptive_information_filename,
                                                     usecols=['PERMNO', 'DATE', 'NAMEENDT', 'SHRCD', 'EXCHCD'],
                                                     parse_dates=['DATE', 'NAMEENDT'])

    logging.info("Renaming the columns to be more descriptive...")
    historical_descriptive_information = (historical_descriptive_information.rename(columns={'PERMNO': 'permanent_number',
                                                                                             'DATE': 'start_date',
                                                                                             'NAMEENDT': 'end_date',
                                                                                             'SHRCD': 'share_code',
                                                                                             'EXCHCD': 'exchange_code'}))

    logging.info("Merging the monthly stock file and historical descriptive information dataframes...")
    crsp_monthly = pd.merge(monthly_stock_file,
                            historical_descriptive_information,
                            how='left',
                            on='permanent_number')

    logging.info("Filtering the CRSP monthly data such that the dates are in the correct range...")
    crsp_monthly = crsp_monthly[(crsp_monthly['end_date'] >= crsp_monthly['month_end_date']) &
                                (crsp_monthly['month_end_date'] >= crsp_monthly['start_date']) &
                                (crsp_monthly['month_end_date'] >= '01/01/1959') &
                                (crsp_monthly['month_end_date'] <= '12/31/2023')]

    logging.info("Filtering the CRSP monthly data to only include American common stocks listed on the NYSE, AMEX, and NASDAQ exchanges...")
    crsp_monthly = crsp_monthly[(crsp_monthly['share_code'].isin([10, 11])) &
                                (crsp_monthly['exchange_code'].isin([1, 2, 3])) &
                                (crsp_monthly['US_incorporation_flag'] == 'Y')]

    logging.info("Keeping only the essential columns for the CRSP monthly data...")
    crsp_monthly = crsp_monthly[['permanent_number', 'permanent_company_number', 'month_end_date', 'monthly_price', 'monthly_return', 'shares_outstanding', 'exchange_code']]

    logging.info("Handling delistings...")
    crsp = handle_delistings(input_delisting_information_filename=input_delisting_information_filename,
                             crsp_monthly=crsp_monthly)
    
    logging.info("Calculating the market equity...")
    crsp['market_equity'] = crsp['monthly_price'] * crsp['shares_outstanding']

    logging.info("Dropping the shares outstanding column...")
    crsp = crsp.drop(['shares_outstanding'], axis=1)

    logging.info("Handling the permanent company number...")
    crsp = handle_permanent_company_number(crsp=crsp)

    logging.info("Sorting the CRSP data by company and then date...")
    crsp = crsp.sort_values(by=['permanent_number', 'month_end_date'])

    logging.info("Creating a column for the weight, which is the market equity from the previous month...")
    crsp['weight'] = crsp.groupby(['permanent_number'])['market_equity'].shift(1)

    logging.info("Creating a column for the Fama-French date, which is six months before the month-end date...")
    crsp['fama_french_date'] = crsp['month_end_date'] + pd.DateOffset(months=-6)

    logging.info("Creating a year column for the Fama-French year...")
    crsp['fama_french_year'] = crsp['fama_french_date'].dt.year

    logging.info("Keeping only the essential columns for the CRSP data...")
    crsp = crsp[['permanent_number', 'month_end_date', 'exchange_code', 'delisting_adjusted_monthly_return', 'market_equity', 'weight', 'fama_french_year', 'monthly_price']]

    logging.info("Saving the processed CRSP data to a CSV file...")
    crsp.to_csv(output_crsp_filename, index=False)

    logging.info("Creating a year column for the CRSP data...")
    crsp['year'] = crsp['month_end_date'].dt.year

    logging.info("Creating a month column for the CRSP data...")
    crsp['month'] = crsp['month_end_date'].dt.month

    logging.info("Creating a dataframe with only the December data...")
    december_market_equity = crsp[crsp['month'] == 12]

    logging.info("Keeping only the essential columns and renaming the market equity column to December market equity...")
    december_market_equity = december_market_equity[['permanent_number', 'market_equity', 'year']].rename(columns={'market_equity': 'december_market_equity'})

    logging.info("Creating a year column for the December market equity...")
    december_market_equity['year'] = december_market_equity['year'] + 1

    logging.info("Creating a dataframe with only the June data...")
    crsp_june = crsp[crsp['month'] == 6]

    logging.info("Merging the June data with the December data...")
    crsp_june = pd.merge(crsp_june,
                         december_market_equity,
                         how='inner',
                         on=['permanent_number', 'year'])
    
    logging.info("Keeping only the essential columns for the CRSP data...")
    crsp_june = crsp_june[['permanent_number', 'month_end_date', 'exchange_code', 'delisting_adjusted_monthly_return', 'market_equity', 'weight', 'december_market_equity']]

    logging.info("Sorting the CRSP June data by company and then date, dropping any duplicates...")
    crsp_june = crsp_june.sort_values(by=['permanent_number', 'month_end_date']).drop_duplicates()

    logging.info("Saving the processed data to a CSV file...")
    crsp_june.to_csv(output_crsp_june_filename, index=False)


def process_ccm_data(input_compustat_filename: str,
                     input_crsp_june_filename: str,
                     input_linking_table_filename: str,
                     output_ccm_filename: str,
                     logging_enabled: bool = True):
    """
    Process the CCM data.

    Parameters:
        input_compustat_filename (str): The file path to the processed Compustat data.
        input_crsp_june_filename (str): The file path to the processed CRSP June data.
        input_linking_table_filename (str): The file path to the raw CRSP Compustat linking table data.
        output_ccm_filename (str): The file path to save the processed CCM data.
        logging_enabled (bool): A boolean indicating whether logging is enabled.

    Returns:
        None
    """
    
    # Set up logging.
    setup_logging(logging_enabled)

    logging.info("Reading in the processed Compustat and CRSP data and parsing the date columns...")
    compustat = pd.read_csv(filepath_or_buffer=input_compustat_filename,
                            parse_dates=['month_end_date'])
    crsp_june = pd.read_csv(filepath_or_buffer=input_crsp_june_filename,
                            parse_dates=['month_end_date'])

    logging.info("Reading in the raw CRSP Compustat linking table data, filtering for the relevant columns to save memory, and parsing the date columns...")
    linking_table = pd.read_csv(filepath_or_buffer=input_linking_table_filename,
                                usecols=['gvkey', 'LPERMNO', 'LINKDT', 'LINKENDDT'],
                                parse_dates=['LINKDT', 'LINKENDDT'])
    
    logging.info("Renaming the columns to be more descriptive...")
    linking_table = linking_table.rename(columns={'gvkey': 'global_company_key',
                                                  'LPERMNO': 'permanent_number',
                                                  'LINKDT': 'link_start_date',
                                                  'LINKENDDT': 'link_end_date'})

    # Securities that have not delisted are given an end date of 'E', so we will replace this with today's date.
    logging.info("Replacing the 'E' values with NaN...")
    linking_table['link_end_date'] = linking_table['link_end_date'].replace('E', np.nan)

    logging.info("Filling the missing values in the link end date column with today's date...")
    linking_table['link_end_date'] = linking_table['link_end_date'].fillna(pd.to_datetime('today'))

    logging.info("Converting the link end date column to a datetime object...")
    linking_table['link_end_date'] = pd.to_datetime(linking_table['link_end_date'])

    logging.info("Merging the relevant Compustat data with the linking table data...")
    linking_table = pd.merge(compustat,
                             linking_table,
                             how='left',
                             on='global_company_key')

    logging.info("Creating a year end column for the linking table data...")
    linking_table['year_end'] = linking_table['month_end_date'] + pd.offsets.YearEnd(0)

    logging.info("Creating a june date column that is six months after the year end date...")
    linking_table['june_date'] = linking_table['year_end'] + pd.offsets.MonthEnd(6)

    logging.info("Filtering the linking table data by date to only include the relevant data...")
    linking_table = linking_table[(linking_table['june_date'] >= linking_table['link_start_date']) &
                                  (linking_table['june_date'] <= linking_table['link_end_date'])]
    
    logging.info("Dropping the unneeded date columns...")
    linking_table = linking_table.drop(['link_start_date', 'link_end_date', 'year_end', 'month_end_date'], axis=1)
        
    logging.info("Merging the CRSP June data with the linking table data...")
    crsp_compustat_june = pd.merge(crsp_june,
                                   linking_table,
                                   how='inner',
                                   left_on=['permanent_number', 'month_end_date'],
                                   right_on=['permanent_number', 'june_date'])
        
    logging.info("Saving the processed CCM June data to a CSV file...")
    crsp_compustat_june.to_csv(output_ccm_filename, index=False)


def process_data():
    """
    Process the Compustat, CRSP, and CCM data.

    Parameters:
        None

    Returns:
        None
    """

    process_compustat_data(input_fundamentals_annual_filename='data/raw_data/raw_compustat_fundamentals_annual.csv',
                           output_compustat_filename='data/processed_data/compustat.csv',
                           additional_columns=[],
                           logging_enabled=True)

    process_crsp_data(input_monthly_stock_file_filename='data/raw_data/raw_crsp_monthly_stock_files.csv',
                      input_historical_descriptive_information_filename='data/raw_data/raw_crsp_historical_descriptive_information.csv',
                      input_delisting_information_filename='data/raw_data/raw_crsp_delisting_information.csv',
                      output_crsp_filename='data/processed_data/crsp.csv',
                      output_crsp_june_filename='data/processed_data/crsp_june.csv')

    process_ccm_data(input_compustat_filename='data/processed_data/compustat.csv',
                     input_crsp_june_filename='data/processed_data/crsp_june.csv',
                     input_linking_table_filename='data/raw_data/raw_crsp_compustat_linking_table.csv',
                     output_ccm_filename='data/processed_data/ccm.csv',
                     logging_enabled=True)
