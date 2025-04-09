import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

def count_multi_college_projects(dataframes: Dict[str, pd.DataFrame]) -> int:
    """
    Count projects involving more than one college.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Number of multi-college projects
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return 0
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['grant_code', 'fiscal_year', 'collegeunit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return 0
    
    # Group by grant_code and fiscal_year
    grouped = df.groupby(['grant_code', 'fiscal_year'])
    
    # Count projects with multiple colleges
    multi_college_count = 0
    
    for _, group in grouped:
        # Get unique colleges for this project-year
        unique_colleges = group['collegeunit'].dropna().unique()
        
        # If there's more than one college, count it
        if len(unique_colleges) > 1:
            multi_college_count += 1
    
    return multi_college_count

def count_multi_co_pi_projects(dataframes: Dict[str, pd.DataFrame]) -> int:
    """
    Count projects with more than one Co-PI.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Number of projects with multiple Co-PIs
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return 0
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['grant_code', 'fiscal_year', 'co_pi_list']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return 0
    
    # Group by grant_code and fiscal_year
    grouped = df.groupby(['grant_code', 'fiscal_year'])
    
    # Count projects with multiple Co-PIs
    multi_co_pi_count = 0
    
    for _, group in grouped:
        # Get all Co-PIs for this project-year
        all_co_pis = []
        for co_pi_list in group['co_pi_list']:
            if isinstance(co_pi_list, list):
                all_co_pis.extend(co_pi_list)
        
        # If there's more than one Co-PI, count it
        if len(all_co_pis) > 1:
            multi_co_pi_count += 1
    
    return multi_co_pi_count

def get_colleges_with_most_collaborators(dataframes: Dict[str, pd.DataFrame], top_n: int = 10) -> pd.DataFrame:
    """
    Identify colleges with the most unique collaborators (PIs and Co-PIs).
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        top_n: Number of top colleges to return
        
    Returns:
        DataFrame with college names and collaborator counts
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return pd.DataFrame(columns=['collegeunit', 'collaborator_count'])
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['collegeunit', 'pi', 'co_pi_list']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame(columns=['collegeunit', 'collaborator_count'])
    
    # Create a dictionary to store collaborators per college
    college_collaborators = {}
    
    # Iterate through each row
    for _, row in df.iterrows():
        college = row['collegeunit']
        
        if pd.isna(college):
            continue
        
        # Initialize the set for this college if it doesn't exist
        if college not in college_collaborators:
            college_collaborators[college] = set()
        
        # Add PI to collaborators
        if not pd.isna(row['pi']):
            college_collaborators[college].add(row['pi'])
        
        # Add Co-PIs to collaborators
        if isinstance(row['co_pi_list'], list):
            for co_pi in row['co_pi_list']:
                if co_pi and not pd.isna(co_pi):
                    college_collaborators[college].add(co_pi)
    
    # Convert to DataFrame
    result = pd.DataFrame([
        {'collegeunit': college, 'collaborator_count': len(collaborators)}
        for college, collaborators in college_collaborators.items()
    ])
    
    if result.empty:
        return result
    
    # Sort by collaborator count in descending order
    result = result.sort_values('collaborator_count', ascending=False)
    
    # Return top N colleges
    return result.head(top_n)

def get_project_counts_by_person(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Count projects per PI and Co-PI.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        DataFrame with person names and project counts
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return pd.DataFrame(columns=['person', 'role', 'project_count'])
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['grant_code', 'fiscal_year', 'pi', 'co_pi_list']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame(columns=['person', 'role', 'project_count'])
    
    # Create dictionaries to store project counts
    pi_projects = {}
    co_pi_projects = {}
    
    # Iterate through each row
    for _, row in df.iterrows():
        grant_code = row['grant_code']
        fiscal_year = row['fiscal_year']
        project_key = f"{grant_code}_{fiscal_year}"
        
        # Count PI projects
        if not pd.isna(row['pi']):
            pi = row['pi']
            if pi not in pi_projects:
                pi_projects[pi] = set()
            pi_projects[pi].add(project_key)
        
        # Count Co-PI projects
        if isinstance(row['co_pi_list'], list):
            for co_pi in row['co_pi_list']:
                if co_pi and not pd.isna(co_pi):
                    if co_pi not in co_pi_projects:
                        co_pi_projects[co_pi] = set()
                    co_pi_projects[co_pi].add(project_key)
    
    # Combine results
    result = []
    
    for pi, projects in pi_projects.items():
        result.append({
            'person': pi,
            'role': 'PI',
            'project_count': len(projects)
        })
    
    for co_pi, projects in co_pi_projects.items():
        result.append({
            'person': co_pi,
            'role': 'Co-PI',
            'project_count': len(projects)
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result)
    
    if result_df.empty:
        return result_df
    
    # Sort by project count in descending order
    result_df = result_df.sort_values('project_count', ascending=False)
    
    return result_df

def get_project_counts_by_college(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Count projects per college.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        DataFrame with college names and project counts
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return pd.DataFrame(columns=['collegeunit', 'project_count'])
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['grant_code', 'fiscal_year', 'collegeunit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame(columns=['collegeunit', 'project_count'])
    
    # Group by college and count unique project-years
    result = df.groupby('collegeunit').apply(
        lambda x: len(x.groupby(['grant_code', 'fiscal_year']).size())
    ).reset_index()
    
    result.columns = ['collegeunit', 'project_count']
    
    if result.empty:
        return result
    
    # Sort by project count in descending order
    result = result.sort_values('project_count', ascending=False)
    
    return result

def get_summary_counts(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    Get total counts of unique PIs, Co-PIs, and colleges.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Dictionary with summary counts
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return {
            'unique_pis': 0,
            'unique_co_pis': 0,
            'unique_colleges': 0
        }
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['pi', 'co_pi_list', 'collegeunit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return {
            'unique_pis': 0,
            'unique_co_pis': 0,
            'unique_colleges': 0
        }
    
    # Count unique PIs
    unique_pis = df['pi'].dropna().nunique()
    
    # Count unique Co-PIs
    unique_co_pis = set()
    for co_pi_list in df['co_pi_list']:
        if isinstance(co_pi_list, list):
            for co_pi in co_pi_list:
                if co_pi and not pd.isna(co_pi):
                    unique_co_pis.add(co_pi)
    
    # Count unique colleges
    unique_colleges = df['collegeunit'].dropna().nunique()
    
    return {
        'unique_pis': unique_pis,
        'unique_co_pis': len(unique_co_pis),
        'unique_colleges': unique_colleges
    }

def filter_by_fiscal_year(dataframes: Dict[str, pd.DataFrame], fiscal_year: int) -> Dict[str, pd.DataFrame]:
    """
    Filter DataFrames by a specific fiscal year.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        fiscal_year: Fiscal year to filter by
        
    Returns:
        Dictionary mapping sheet names to filtered DataFrames
    """
    filtered_dfs = {}
    
    for sheet_name, df in dataframes.items():
        if 'fiscal_year' in df.columns:
            filtered_dfs[sheet_name] = df[df['fiscal_year'] == fiscal_year].copy()
        else:
            filtered_dfs[sheet_name] = df.copy()
    
    return filtered_dfs

def filter_by_fiscal_year_range(dataframes: Dict[str, pd.DataFrame], start_year: int, end_year: int) -> Dict[str, pd.DataFrame]:
    """
    Filter DataFrames by a range of fiscal years.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        start_year: Start fiscal year (inclusive)
        end_year: End fiscal year (inclusive)
        
    Returns:
        Dictionary mapping sheet names to filtered DataFrames
    """
    filtered_dfs = {}
    
    for sheet_name, df in dataframes.items():
        if 'fiscal_year' in df.columns:
            mask = (df['fiscal_year'] >= start_year) & (df['fiscal_year'] <= end_year)
            filtered_dfs[sheet_name] = df[mask].copy()
        else:
            filtered_dfs[sheet_name] = df.copy()
    
    return filtered_dfs

def get_projects_by_co_pi_count(dataframes: Dict[str, pd.DataFrame], condition: str, count: int) -> pd.DataFrame:
    """
    Get projects based on a condition related to the number of Co-PIs.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        condition: One of 'exactly', 'more_than', 'less_than', 'at_least', 'at_most'
        count: Number of Co-PIs to compare against
        
    Returns:
        DataFrame containing projects that match the specified condition
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return pd.DataFrame()
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['grant_code', 'fiscal_year', 'pi', 'co_pi_list']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Group by grant_code and fiscal_year to get unique projects
    projects = []
    
    for (grant_code, fiscal_year), group in df.groupby(['grant_code', 'fiscal_year']):
        # Get unique PIs for this project
        pis = group['pi'].dropna().unique().tolist()
        
        # Get all Co-PIs for this project
        all_co_pis = set()
        for co_pi_list in group['co_pi_list']:
            if isinstance(co_pi_list, list):
                all_co_pis.update([cp for cp in co_pi_list if cp and not pd.isna(cp)])
        
        # Convert to list and get count
        co_pis = list(all_co_pis)
        co_pi_count = len(co_pis)
        
        # Check if this project satisfies the condition
        include_project = False
        if condition == 'exactly' and co_pi_count == count:
            include_project = True
        elif condition == 'more_than' and co_pi_count > count:
            include_project = True
        elif condition == 'less_than' and co_pi_count < count:
            include_project = True
        elif condition == 'at_least' and co_pi_count >= count:
            include_project = True
        elif condition == 'at_most' and co_pi_count <= count:
            include_project = True
        
        if include_project:
            # Get college units for this project
            college_units = group['collegeunit'].dropna().unique().tolist()
            
            # Get award amount if available
            award_amount = group['award_amount'].sum() if 'award_amount' in group.columns else None
            
            # Add project details to list
            projects.append({
                'grant_code': grant_code,
                'fiscal_year': fiscal_year,
                'pis': pis,
                'co_pis': co_pis,
                'co_pi_count': co_pi_count,
                'college_units': college_units,
                'award_amount': award_amount
            })
    
    # Convert to DataFrame
    result = pd.DataFrame(projects)
    
    return result

def get_projects_by_co_pi_count_yearly(dataframes: Dict[str, pd.DataFrame], condition: str, count: int) -> pd.DataFrame:
    """
    Get counts of projects based on Co-PI conditions, grouped by fiscal year.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        condition: One of 'exactly', 'more_than', 'less_than', 'at_least', 'at_most'
        count: Number of Co-PIs to compare against
        
    Returns:
        DataFrame with yearly statistics (fiscal_year, matching_projects, total_projects, etc.)
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return pd.DataFrame()
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['grant_code', 'fiscal_year', 'pi', 'co_pi_list']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Get all fiscal years
    fiscal_years = sorted(df['fiscal_year'].dropna().unique())
    
    # Initialize results
    yearly_stats = []
    
    # Calculate statistics for each fiscal year
    for year in fiscal_years:
        # Filter data for this year
        year_df = df[df['fiscal_year'] == year]
        
        # Get unique projects for this year
        unique_projects = year_df.groupby(['grant_code', 'fiscal_year'])
        total_projects = unique_projects.ngroups
        
        # Count projects matching the Co-PI condition
        matching_projects = 0
        total_pis = 0
        total_co_pis = 0
        
        for (grant_code, _), group in unique_projects:
            # Count unique PIs
            unique_pis = group['pi'].dropna().nunique()
            total_pis += unique_pis
            
            # Get all Co-PIs for this project
            all_co_pis = set()
            for co_pi_list in group['co_pi_list']:
                if isinstance(co_pi_list, list):
                    all_co_pis.update([cp for cp in co_pi_list if cp and not pd.isna(cp)])
            
            # Get Co-PI count
            co_pi_count = len(all_co_pis)
            total_co_pis += co_pi_count
            
            # Check if this project satisfies the condition
            if condition == 'exactly' and co_pi_count == count:
                matching_projects += 1
            elif condition == 'more_than' and co_pi_count > count:
                matching_projects += 1
            elif condition == 'less_than' and co_pi_count < count:
                matching_projects += 1
            elif condition == 'at_least' and co_pi_count >= count:
                matching_projects += 1
            elif condition == 'at_most' and co_pi_count <= count:
                matching_projects += 1
        
        # Add statistics for this year
        yearly_stats.append({
            'fiscal_year': year,
            'matching_projects': matching_projects,
            'total_projects': total_projects,
            'total_pis': total_pis,
            'total_co_pis': total_co_pis,
            'matching_percentage': (matching_projects / total_projects * 100) if total_projects > 0 else 0
        })
    
    # Convert to DataFrame
    result = pd.DataFrame(yearly_stats)
    
    return result

def get_project_details_by_co_pi_filter(dataframes: Dict[str, pd.DataFrame], fiscal_year: int, 
                                       condition: str, count: int) -> pd.DataFrame:
    """
    Get detailed information about projects that match the Co-PI filter for a specific year.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        fiscal_year: The fiscal year to filter by
        condition: One of 'exactly', 'more_than', 'less_than', 'at_least', 'at_most'
        count: Number of Co-PIs to compare against
        
    Returns:
        DataFrame with detailed project information
    """
    # Filter by fiscal year first
    year_dataframes = filter_by_fiscal_year(dataframes, fiscal_year)
    
    # Get projects matching the Co-PI condition
    projects = get_projects_by_co_pi_count(year_dataframes, condition, count)
    
    if projects.empty:
        return projects
    
    # Expand the projects DataFrame for better display
    detailed_rows = []
    
    for _, project in projects.iterrows():
        # Basic project info
        basic_info = {
            'grant_code': project['grant_code'],
            'fiscal_year': project['fiscal_year'],
            'co_pi_count': project['co_pi_count'],
            'award_amount': project['award_amount']
        }
        
        # Add PIs
        for i, pi in enumerate(project['pis']):
            pi_key = f'pi_{i+1}'
            basic_info[pi_key] = pi
        
        # Add Co-PIs
        for i, co_pi in enumerate(project['co_pis']):
            co_pi_key = f'co_pi_{i+1}'
            basic_info[co_pi_key] = co_pi
        
        # Add College Units
        for i, unit in enumerate(project['college_units']):
            unit_key = f'college_unit_{i+1}'
            basic_info[unit_key] = unit
        
        detailed_rows.append(basic_info)
    
    # Convert to DataFrame
    detailed_df = pd.DataFrame(detailed_rows)
    
    return detailed_df 