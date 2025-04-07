import pandas as pd
import os
import re
from typing import Dict, List, Tuple, Optional, Union, Any

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def export_to_excel(dataframes: Dict[str, pd.DataFrame], output_path: str) -> None:
    """
    Export multiple DataFrames to an Excel file.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        output_path: Path to save the Excel file
    """
    # Ensure the directory exists
    directory = os.path.dirname(output_path)
    ensure_directory_exists(directory)
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def export_to_csv(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    Export a DataFrame to a CSV file.
    
    Args:
        dataframe: DataFrame to export
        output_path: Path to save the CSV file
    """
    # Ensure the directory exists
    directory = os.path.dirname(output_path)
    ensure_directory_exists(directory)
    
    # Export to CSV
    dataframe.to_csv(output_path, index=False)

def standardize_name(name: str) -> str:
    """
    Standardize a name format (e.g., "Smith, John" or "John Smith").
    
    Args:
        name: Name to standardize
        
    Returns:
        Standardized name
    """
    if pd.isna(name):
        return ""
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Check if name is in "Last, First" format
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            last_name = parts[0].strip()
            first_name = parts[1].strip()
            return f"{last_name}, {first_name}"
    
    # If not in "Last, First" format, assume it's already standardized
    return name

def extract_project_year(grant_code: str, fiscal_year: int) -> str:
    """
    Create a unique project-year identifier.
    
    Args:
        grant_code: Grant code
        fiscal_year: Fiscal year
        
    Returns:
        Project-year identifier
    """
    return f"{grant_code}_{fiscal_year}"

def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """
    Format a number as a percentage.
    
    Args:
        value: Value to format
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.1f}%" 