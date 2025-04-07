import pandas as pd
import re
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column headers by converting to lowercase and replacing spaces with underscores.
    
    Args:
        df: DataFrame to normalize
        
    Returns:
        DataFrame with normalized headers
    """
    logger.info("Normalizing DataFrame headers")
    logger.debug(f"Original columns: {list(df.columns)}")
    
    # Create a mapping of original column names to normalized names
    header_mapping = {
        col: col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        for col in df.columns
    }
    
    logger.debug(f"Header mapping: {header_mapping}")
    
    # Rename columns
    df_normalized = df.rename(columns=header_mapping)
    logger.debug(f"Normalized columns: {list(df_normalized.columns)}")
    return df_normalized

def parse_fiscal_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert fiscal years to full year format (e.g., 15 -> 2015).
    
    Args:
        df: DataFrame with fiscal_year column
        
    Returns:
        DataFrame with normalized fiscal years
    """
    logger.info("Parsing fiscal years")
    
    if 'fiscal_year' not in df.columns:
        logger.warning("fiscal_year column not found in DataFrame")
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Log original fiscal years and their types
    original_years = df_copy['fiscal_year'].unique()
    logger.debug(f"Original fiscal years: {original_years}")
    logger.debug(f"Original fiscal year types: {[type(year) for year in original_years]}")
    
    # Convert fiscal years to integers first, handling any string values
    def convert_fiscal_year(x):
        if pd.isna(x):
            return None
        try:
            # Convert to string and clean
            year_str = str(x).strip().replace(',', '')
            # Convert to float then int
            year_int = int(float(year_str))
            # Add 2000 if it's a two-digit year
            if year_int < 100:
                year_int += 2000
            logger.debug(f"Converted {x} ({type(x)}) to {year_int}")
            return year_int
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert fiscal year value: {x} ({type(x)}). Error: {str(e)}")
            return None
    
    df_copy['fiscal_year'] = df_copy['fiscal_year'].apply(convert_fiscal_year)
    
    # Log the fiscal years after conversion
    converted_years = sorted([y for y in df_copy['fiscal_year'].unique() if pd.notnull(y)])
    logger.debug(f"Normalized fiscal years: {converted_years}")
    logger.debug(f"Number of unique fiscal years: {len(converted_years)}")
    
    return df_copy

def split_co_pi_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split Co-PI names that are delimited by '/' or '\'.
    
    Args:
        df: DataFrame with co_pis or co_pi column
        
    Returns:
        DataFrame with split Co-PI names
    """
    logger.info("Splitting Co-PI names")
    co_pi_columns = ['co_pis', 'co_pi']  # Check both possible column names
    found_column = None
    
    for col in co_pi_columns:
        if col in df.columns:
            found_column = col
            logger.info(f"Found Co-PI column: {col}")
            break
    
    if not found_column:
        logger.warning("No Co-PI column found")
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Function to split Co-PI names
    def split_names(name_str):
        if pd.isna(name_str):
            return []
        
        # Split by '/' or '\'
        names = re.split(r'[/\\]', str(name_str))
        
        # Strip whitespace and filter out empty strings
        return [name.strip() for name in names if name.strip()]
    
    # Apply the splitting function
    logger.debug(f"Original Co-PI values (first 5): {df_copy[found_column].head()}")
    df_copy['co_pi_list'] = df_copy[found_column].apply(split_names)
    logger.debug(f"Split Co-PI lists (first 5): {df_copy['co_pi_list'].head()}")
    
    return df_copy

def deduplicate_co_pis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate Co-PIs for the same project-year.
    
    Args:
        df: DataFrame with Co-PI information
        
    Returns:
        DataFrame with deduplicated Co-PIs
    """
    logger.info("Deduplicating Co-PIs")
    
    required_cols = ['co_pi_list', 'grant_code', 'fiscal_year']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns for deduplication: {missing_cols}")
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Group by grant_code and fiscal_year
    grouped = df_copy.groupby(['grant_code', 'fiscal_year'])
    logger.debug(f"Number of groups: {len(grouped)}")
    
    # Function to deduplicate Co-PIs within each group
    def deduplicate_group(group):
        # Get all Co-PIs for this group
        all_co_pis = []
        for co_pi_list in group['co_pi_list']:
            if isinstance(co_pi_list, list):
                all_co_pis.extend(co_pi_list)
        
        # Remove duplicates while preserving order
        unique_co_pis = []
        for co_pi in all_co_pis:
            if co_pi not in unique_co_pis:
                unique_co_pis.append(co_pi)
        
        # Create a new row with the deduplicated list
        result = group.iloc[0].copy()
        result['co_pi_list'] = unique_co_pis
        return result
    
    # Apply the deduplication function to each group and reset the index
    logger.debug("Starting group deduplication")
    df_copy = grouped.apply(deduplicate_group).reset_index(drop=True)
    logger.debug(f"Shape after deduplication: {df_copy.shape}")
    
    return df_copy

def clean_data(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean and normalize all DataFrames.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Dictionary mapping sheet names to cleaned DataFrames
    """
    logger.info("Starting data cleaning process")
    cleaned_dfs = {}
    
    for sheet_name, df in dataframes.items():
        logger.info(f"Cleaning sheet: {sheet_name}")
        logger.debug(f"Initial shape: {df.shape}")
        logger.debug(f"Initial columns: {list(df.columns)}")
        logger.debug(f"Initial data types:\n{df.dtypes}")
        logger.debug(f"Initial missing values:\n{df.isnull().sum()}")
        
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Parse fiscal years
        df_copy = parse_fiscal_years(df_copy)
        logger.debug(f"After parsing fiscal years - shape: {df_copy.shape}")
        
        # Split Co-PI names
        df_copy = split_co_pi_names(df_copy)
        logger.debug(f"After splitting Co-PI names - shape: {df_copy.shape}")
        logger.debug(f"After splitting Co-PI names - columns: {list(df_copy.columns)}")
        
        # Deduplicate Co-PIs if the necessary columns exist
        if 'co_pi_list' in df_copy.columns:
            df_copy = deduplicate_co_pis(df_copy)
            logger.debug(f"After deduplicating Co-PIs - shape: {df_copy.shape}")
        
        cleaned_dfs[sheet_name] = df_copy
        logger.info(f"Finished cleaning sheet: {sheet_name}")
        logger.debug(f"Final shape: {df_copy.shape}")
        logger.debug(f"Final columns: {list(df_copy.columns)}")
        logger.debug(f"Final data types:\n{df_copy.dtypes}")
        logger.debug(f"Final missing values:\n{df_copy.isnull().sum()}")
    
    logger.info("Data cleaning process complete")
    return cleaned_dfs 