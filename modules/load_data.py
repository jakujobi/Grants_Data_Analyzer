import pandas as pd
import os
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def load_excel_file(file_path: str, sheet_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load data from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        sheet_names: Optional list of sheet names to load. If None, loads all sheets.
        
    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    logger.info(f"Loading Excel file: {file_path}")
    logger.info(f"Requested sheets: {sheet_names}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Default sheet names if not provided
    if sheet_names is None:
        sheet_names = ["AwardsRawData", "AwardsCoPIsRawData"]  # Use the actual sheet names
        logger.info(f"Using default sheet names: {sheet_names}")
    
    # Load the Excel file
    try:
        xl = pd.ExcelFile(file_path)
        available_sheets = xl.sheet_names
        logger.info(f"Available sheets in file: {available_sheets}")
    except Exception as e:
        logger.error(f"Error opening Excel file: {str(e)}", exc_info=True)
        raise
    
    # Check if required sheets exist
    missing_sheets = [sheet for sheet in sheet_names if sheet not in available_sheets]
    if missing_sheets:
        logger.warning(f"Missing sheets: {missing_sheets}")
        logger.info(f"Available sheets: {available_sheets}")
        # Use available sheets instead
        sheet_names = [sheet for sheet in sheet_names if sheet in available_sheets]
        logger.info(f"Proceeding with available sheets: {sheet_names}")
    
    # Load each sheet
    dataframes = {}
    for sheet in sheet_names:
        try:
            logger.info(f"Loading sheet: {sheet}")
            df = pd.read_excel(file_path, sheet_name=sheet)
            logger.info(f"Sheet {sheet} loaded successfully")
            logger.debug(f"Sheet {sheet} shape: {df.shape}")
            logger.debug(f"Sheet {sheet} columns: {list(df.columns)}")
            dataframes[sheet] = df
        except Exception as e:
            logger.error(f"Error loading sheet {sheet}: {str(e)}", exc_info=True)
            raise
    
    return dataframes

def validate_dataframes(dataframes: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
    """
    Validate that the loaded DataFrames have the required columns.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    logger.info("Starting DataFrame validation")
    errors = []
    
    # Required columns for each sheet (using original column names)
    required_columns = {
        "AwardsRawData": ["Grant Code", "Fiscal Year", "PI", "Co-PI(s)", "CollegeUnit", "Department"],
        "AwardsCoPIsRawData": ["Grant Code", "Fiscal Year", "PI", "Co-PI", "CollegeUnit", "Department"]
    }
    
    logger.debug(f"Required columns: {required_columns}")
    
    # Check each sheet
    for sheet_name, required_cols in required_columns.items():
        if sheet_name not in dataframes:
            error_msg = f"Missing required sheet: {sheet_name}"
            logger.warning(error_msg)
            errors.append(error_msg)
            continue
        
        df = dataframes[sheet_name]
        logger.debug(f"Validating sheet {sheet_name}")
        logger.debug(f"Available columns: {list(df.columns)}")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            error_msg = f"Sheet '{sheet_name}' is missing required columns: {missing_cols}"
            logger.warning(error_msg)
            logger.debug(f"Available columns in {sheet_name}: {list(df.columns)}")
            errors.append(error_msg)
    
    is_valid = len(errors) == 0
    logger.info(f"Validation complete. Valid: {is_valid}")
    if not is_valid:
        logger.warning(f"Validation errors: {errors}")
    
    return is_valid, errors

def normalize_column_names(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Normalize column names in all DataFrames.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Dictionary mapping sheet names to DataFrames with normalized column names
    """
    logger.info("Starting column name normalization")
    normalized_dfs = {}
    
    # Column name mapping for each sheet
    column_mappings = {
        "AwardsRawData": {
            "Report": "report",
            "CollegeUnit": "collegeunit",
            "Department": "department",
            "Fiscal Year": "fiscal_year",
            "Quarter": "quarter",
            "Organization Code": "org_code",
            "Organization Code Title": "org_title",
            "PI": "pi",
            "Co-PI(s)": "co_pis",
            "Grant Code": "grant_code",
            "Fund": "fund",
            "Fund Title": "fund_title",
            "Project Title": "project_title",
            "Agency": "agency",
            "Fund Source": "fund_source",
            "Sponsor": "sponsor",
            "Grant List Sponsor": "grant_list_sponsor",
            "Program": "program",
            "Program Title": "program_title",
            "Quarter Title": "quarter_title",
            "Month": "month",
            "Period": "period",
            "Award Count": "award_count",
            "Award Amount": "award_amount",
            "F&A Budgeted": "fa_budgeted"
        },
        "AwardsCoPIsRawData": {
            "Report": "report",
            "CollegeUnit": "collegeunit",
            "Department": "department",
            "Fiscal Year": "fiscal_year",
            "Quarter": "quarter",
            "Organization Code": "org_code",
            "Organization Code Title": "org_title",
            "PI": "pi",
            "Co-PI": "co_pi",
            "Grant Code": "grant_code",
            "Project Title": "project_title",
            "Agency": "agency",
            "Fund Source": "fund_source",
            "Sponsor": "sponsor",
            "Grant List Sponsor": "grant_list_sponsor",
            "Program": "program",
            "Program Title": "program_title",
            "Award Amount": "award_amount"
        }
    }
    
    logger.debug(f"Column mappings: {column_mappings}")
    
    for sheet_name, df in dataframes.items():
        logger.info(f"Normalizing columns for sheet: {sheet_name}")
        logger.debug(f"Original columns: {list(df.columns)}")
        
        if sheet_name in column_mappings:
            # Create a copy of the DataFrame
            df_copy = df.copy()
            
            # Rename columns using the mapping
            mapping = column_mappings[sheet_name]
            df_copy = df_copy.rename(columns=mapping)
            
            # For any columns not in the mapping, normalize them
            for col in df_copy.columns:
                if col not in mapping.values():
                    new_col = col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_').replace('&', 'and')
                    df_copy = df_copy.rename(columns={col: new_col})
            
            normalized_dfs[sheet_name] = df_copy
        else:
            # For sheets without a specific mapping, use general normalization
            df_copy = df.copy()
            df_copy.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_').replace('&', 'and')
                             for col in df_copy.columns]
            normalized_dfs[sheet_name] = df_copy
        
        logger.debug(f"Normalized columns: {list(normalized_dfs[sheet_name].columns)}")
    
    logger.info("Column name normalization complete")
    return normalized_dfs

def get_fiscal_years(dataframes: Dict[str, pd.DataFrame]) -> List[int]:
    """
    Extract all unique fiscal years from the data.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        List of unique fiscal years
    """
    logger.info("Extracting fiscal years")
    fiscal_years = set()
    
    for sheet_name, df in dataframes.items():
        if "fiscal_year" in df.columns:
            # Get non-null fiscal years
            years = df["fiscal_year"].dropna()
            logger.debug(f"Raw fiscal years from {sheet_name}: {years.unique()}")
            
            # Convert each year to integer
            for year in years:
                try:
                    # Convert to string first to handle any formatting
                    year_str = str(year).replace(',', '')
                    # Convert to float then int to handle decimal points
                    year_int = int(float(year_str))
                    # Add 2000 if it's a two-digit year
                    if year_int < 100:
                        year_int += 2000
                    fiscal_years.add(year_int)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert fiscal year value: {year}")
                    continue
    
    result = sorted(list(fiscal_years))
    logger.info(f"Found fiscal years: {result}")
    return result

def get_college_units(dataframes: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Extract all unique college units from the data.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        List of unique college units
    """
    logger.info("Extracting college units")
    college_units = set()
    
    for sheet_name, df in dataframes.items():
        if "collegeunit" in df.columns:
            units = df["collegeunit"].dropna().unique()
            logger.debug(f"College units from {sheet_name}: {units}")
            college_units.update(units)
    
    result = sorted(list(college_units))
    logger.info(f"Found college units: {result}")
    return result 