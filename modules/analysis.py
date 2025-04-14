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

def identify_multi_college_projects(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Identifies projects where the PI and Co-PIs come from more than one college.
    
    A project is considered "multi-college" if the PI and Co-PIs collectively represent
    more than one unique college unit.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        DataFrame containing project details with a 'is_multi_college' flag and 'college_count'
    """
    # Add debug info
    print("\n--- Starting multi-college project identification ---")
    
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return pd.DataFrame()
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['grant_code', 'fiscal_year', 'pi', 'co_pi_list', 'collegeunit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Check for co-pi data
    has_copi_sheet = 'AwardsCoPIsRawData' in dataframes
    print(f"Co-PI data sheet available: {has_copi_sheet}")
    
    if has_copi_sheet:
        copi_df = dataframes['AwardsCoPIsRawData']
        print(f"Co-PI data columns: {list(copi_df.columns)}")
    
    # Create a person (PI or Co-PI) to college mapping 
    person_to_college = {}
    
    # First add the PIs to the mapping from the main awards sheet
    for _, row in df.iterrows():
        if not pd.isna(row['pi']) and not pd.isna(row['collegeunit']):
            person_to_college[row['pi']] = row['collegeunit']
    
    # If we have the co-pi sheet, get college info from there
    if has_copi_sheet and 'co_pi' in copi_df.columns and 'collegeunit' in copi_df.columns:
        for _, row in copi_df.iterrows():
            if not pd.isna(row['co_pi']) and not pd.isna(row['collegeunit']):
                person_to_college[row['co_pi']] = row['collegeunit']
    
    print(f"Created person-to-college mapping with {len(person_to_college)} entries")
    
    # Initialize list to collect project data
    project_data = []
    
    # Group by unique projects (grant_code + fiscal_year)
    for (grant_code, fiscal_year), project_group in df.groupby(['grant_code', 'fiscal_year']):
        # Get all colleges associated with PIs and Co-PIs for this project
        colleges = set()
        
        # Add PI college(s) directly from the project group
        pi_names = project_group['pi'].dropna().unique()
        pi_colleges = project_group['collegeunit'].dropna().unique()
        
        for college in pi_colleges:
            if college and not pd.isna(college):
                colleges.add(college)
        
        # Collect all co-pis from the main sheet
        co_pis = []
        for co_pi_list in project_group['co_pi_list']:
            if isinstance(co_pi_list, list):
                co_pis.extend([cp for cp in co_pi_list if cp and not pd.isna(cp)])
        
        # Try to find colleges for Co-PIs using the person-to-college mapping
        for co_pi in co_pis:
            if co_pi in person_to_college:
                co_pi_college = person_to_college[co_pi]
                if co_pi_college and not pd.isna(co_pi_college):
                    colleges.add(co_pi_college)
        
        # Also check Co-PI sheet directly for this grant code
        if has_copi_sheet and 'grant_code' in copi_df.columns:
            grant_copis = copi_df[copi_df['grant_code'] == grant_code]
            
            # If there's co-pi college data for this grant, add those colleges
            if not grant_copis.empty and 'collegeunit' in grant_copis.columns:
                for _, row in grant_copis.iterrows():
                    if not pd.isna(row['collegeunit']):
                        colleges.add(row['collegeunit'])
            
            # Also add any Co-PIs from this data that we didn't already have
            if not grant_copis.empty and 'co_pi' in grant_copis.columns:
                for _, row in grant_copis.iterrows():
                    if not pd.isna(row['co_pi']) and row['co_pi'] not in co_pis:
                        co_pis.append(row['co_pi'])
        
        # Determine if this is a multi-college project
        college_count = len(colleges)
        is_multi_college = college_count > 1
        
        # Add project to the list
        project_data.append({
            'grant_code': grant_code,
            'fiscal_year': fiscal_year,
            'pis': pi_names.tolist(),
            'co_pis': co_pis,
            'college_units': list(colleges),
            'college_count': college_count,
            'is_multi_college': is_multi_college,
            'award_amount': project_group['award_amount'].sum() if 'award_amount' in project_group.columns else None
        })
    
    result_df = pd.DataFrame(project_data)
    
    # Debug info - print distribution of college counts
    if not result_df.empty:
        counts = result_df['college_count'].value_counts().sort_index()
        print("\nDistribution of college counts in projects:")
        for count, num_projects in counts.items():
            print(f"  {count} college(s): {num_projects} projects")
        
        multi_count = result_df['is_multi_college'].sum()
        total = len(result_df)
        print(f"Multi-college projects: {multi_count} out of {total} ({multi_count/total*100:.1f}%)")
        
        # Show a few examples of multi-college projects
        if multi_count > 0:
            print("\nExample multi-college projects:")
            examples = result_df[result_df['is_multi_college']].head(3)
            for _, project in examples.iterrows():
                print(f"  Grant: {project['grant_code']}, Year: {project['fiscal_year']}")
                print(f"  PIs: {project['pis']}")
                print(f"  Co-PIs: {project['co_pis']}")
                print(f"  Colleges: {project['college_units']}")
                print()
    else:
        print("No project data found")
    
    print("--- Finished multi-college project identification ---\n")
    return result_df

def get_multi_college_project_yearly_stats(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate yearly statistics for multi-college projects.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        DataFrame with yearly statistics for multi-college projects
    """
    # Get project data
    project_data = identify_multi_college_projects(dataframes)
    
    if project_data.empty:
        return pd.DataFrame()
    
    # Group by fiscal year and calculate metrics
    yearly_stats = []
    
    for fiscal_year, year_group in project_data.groupby('fiscal_year'):
        # Count projects
        total_projects = len(year_group)
        multi_college_projects = year_group['is_multi_college'].sum()
        single_college_projects = total_projects - multi_college_projects
        
        # Calculate percentages
        multi_college_percentage = (multi_college_projects / total_projects * 100) if total_projects > 0 else 0
        
        # Calculate average colleges per project
        avg_colleges_all = year_group['college_count'].mean()
        avg_colleges_multi = year_group[year_group['is_multi_college']]['college_count'].mean() if multi_college_projects > 0 else 0
        
        # Calculate average award amounts if available
        avg_award_all = None
        avg_award_multi = None
        avg_award_single = None
        
        if 'award_amount' in year_group.columns:
            # Filter to non-null awards
            valid_awards = year_group.dropna(subset=['award_amount'])
            if not valid_awards.empty:
                avg_award_all = valid_awards['award_amount'].mean()
            
            valid_multi = year_group[year_group['is_multi_college']].dropna(subset=['award_amount'])
            if not valid_multi.empty and multi_college_projects > 0:
                avg_award_multi = valid_multi['award_amount'].mean()
            
            valid_single = year_group[~year_group['is_multi_college']].dropna(subset=['award_amount'])
            if not valid_single.empty and single_college_projects > 0:
                avg_award_single = valid_single['award_amount'].mean()
        
        # Add statistics for this year
        yearly_stats.append({
            'fiscal_year': fiscal_year,
            'total_projects': total_projects,
            'multi_college_projects': multi_college_projects,
            'single_college_projects': single_college_projects,
            'multi_college_percentage': multi_college_percentage,
            'avg_colleges_per_project': avg_colleges_all,
            'avg_colleges_per_multi_project': avg_colleges_multi,
            'avg_award_all_projects': avg_award_all,
            'avg_award_multi_projects': avg_award_multi,
            'avg_award_single_projects': avg_award_single
        })
    
    # Convert to DataFrame and sort by fiscal year
    result = pd.DataFrame(yearly_stats).sort_values('fiscal_year')
    
    return result

def get_college_collaboration_metrics(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate collaboration metrics for each college.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        DataFrame with collaboration metrics for each college
    """
    # Get project data
    project_data = identify_multi_college_projects(dataframes)
    
    if project_data.empty:
        return pd.DataFrame()
    
    # Initialize metrics
    college_metrics = {}
    
    # Analyze each project
    for _, project in project_data.iterrows():
        colleges = project['college_units']
        
        # Skip if no college information
        if not colleges:
            continue
        
        # Initialize college entries if needed
        for college in colleges:
            if college not in college_metrics:
                college_metrics[college] = {
                    'college': college,
                    'total_projects': 0,
                    'multi_college_projects': 0,
                    'collaboration_partners': set(),
                    'pi_count': 0,
                    'co_pi_count': 0
                }
        
        # Update metrics for each college in this project
        for college in colleges:
            college_metrics[college]['total_projects'] += 1
            
            # Update multi-college project count
            if project['is_multi_college']:
                college_metrics[college]['multi_college_projects'] += 1
                
                # Update collaboration partners
                for partner in colleges:
                    if partner != college:
                        college_metrics[college]['collaboration_partners'].add(partner)
            
            # Update PI and Co-PI counts
            # This is a simplification - we don't have exact mapping of PI/Co-PI to college
            college_metrics[college]['pi_count'] += len(project['pis'])
            college_metrics[college]['co_pi_count'] += len(project['co_pis'])
    
    # Convert to list of dictionaries
    metrics_list = []
    for college, metrics in college_metrics.items():
        metrics_list.append({
            'college': metrics['college'],
            'total_projects': metrics['total_projects'],
            'multi_college_projects': metrics['multi_college_projects'],
            'multi_college_percentage': (metrics['multi_college_projects'] / metrics['total_projects'] * 100) 
                                       if metrics['total_projects'] > 0 else 0,
            'collaboration_partner_count': len(metrics['collaboration_partners']),
            'collaboration_partners': list(metrics['collaboration_partners']),
            'pi_count': metrics['pi_count'],
            'co_pi_count': metrics['co_pi_count']
        })
    
    return pd.DataFrame(metrics_list).sort_values('multi_college_projects', ascending=False)

def get_multi_college_vs_single_college_comparison(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Compare metrics between multi-college and single-college projects.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Dictionary with comparison metrics
    """
    # Get project data
    project_data = identify_multi_college_projects(dataframes)
    
    if project_data.empty:
        return {
            'total_projects': 0,
            'multi_college_projects': 0,
            'single_college_projects': 0,
            'multi_college_percentage': 0,
            'avg_colleges_multi': 0,
            'avg_award_multi': None,
            'avg_award_single': None,
            'award_difference': None,
            'award_difference_percentage': None
        }
    
    # Count projects
    total_projects = len(project_data)
    multi_college_projects = project_data['is_multi_college'].sum()
    single_college_projects = total_projects - multi_college_projects
    
    # Calculate percentages
    multi_college_percentage = (multi_college_projects / total_projects * 100) if total_projects > 0 else 0
    
    # Calculate average colleges per multi-college project
    avg_colleges_multi = project_data[project_data['is_multi_college']]['college_count'].mean() if multi_college_projects > 0 else 0
    
    # Calculate average award amounts
    avg_award_multi = None
    avg_award_single = None
    award_difference = None
    award_difference_percentage = None
    
    if 'award_amount' in project_data.columns:
        multi_college_df = project_data[project_data['is_multi_college']].dropna(subset=['award_amount'])
        single_college_df = project_data[~project_data['is_multi_college']].dropna(subset=['award_amount'])
        
        if not multi_college_df.empty and multi_college_projects > 0:
            avg_award_multi = multi_college_df['award_amount'].mean()
        
        if not single_college_df.empty and single_college_projects > 0:
            avg_award_single = single_college_df['award_amount'].mean()
        
        if avg_award_multi is not None and avg_award_single is not None:
            award_difference = avg_award_multi - avg_award_single
            award_difference_percentage = (award_difference / avg_award_single * 100) if avg_award_single > 0 else 0
    
    return {
        'total_projects': total_projects,
        'multi_college_projects': multi_college_projects,
        'single_college_projects': single_college_projects,
        'multi_college_percentage': multi_college_percentage,
        'avg_colleges_multi': avg_colleges_multi,
        'avg_award_multi': avg_award_multi,
        'avg_award_single': avg_award_single,
        'award_difference': award_difference,
        'award_difference_percentage': award_difference_percentage
    }

def get_multi_college_projects_by_year(dataframes: Dict[str, pd.DataFrame], fiscal_year: int) -> pd.DataFrame:
    """
    Get multi-college projects for a specific fiscal year.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        fiscal_year: The fiscal year to filter by
        
    Returns:
        DataFrame containing multi-college projects for the specified year
    """
    # Get all project data
    project_data = identify_multi_college_projects(dataframes)
    
    if project_data.empty:
        return pd.DataFrame()
    
    # Filter by fiscal year and multi-college flag
    filtered_data = project_data[(project_data['fiscal_year'] == fiscal_year) & 
                                (project_data['is_multi_college'])]
    
    return filtered_data

def analyze_college_collaboration_network(dataframes: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Analyze the network of college collaborations and create a pairwise collaboration matrix.
    
    This function identifies how frequently different colleges collaborate with each other
    across projects by constructing a matrix of pairwise relationships.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Tuple containing:
        - DataFrame with the collaboration matrix (colleges as both rows and columns)
        - Dictionary mapping each college pair to the list of their collaborative projects
    """
    # Get project data
    project_data = identify_multi_college_projects(dataframes)
    
    if project_data.empty:
        return pd.DataFrame(), {}
    
    # Filter to only multi-college projects
    multi_college_projects = project_data[project_data['is_multi_college']]
    
    if multi_college_projects.empty:
        return pd.DataFrame(), {}
    
    # Get all unique colleges
    all_colleges = set()
    for colleges in multi_college_projects['college_units']:
        all_colleges.update(colleges)
    
    all_colleges = sorted(list(all_colleges))
    
    # Initialize collaboration matrix with zeros
    collaboration_matrix = pd.DataFrame(
        0, 
        index=all_colleges, 
        columns=all_colleges
    )
    
    # Initialize project tracking dictionary
    college_pair_projects = {}
    
    # Count collaborations between each pair of colleges
    for _, project in multi_college_projects.iterrows():
        colleges = project['college_units']
        
        # Skip if less than 2 colleges (shouldn't happen with multi-college projects)
        if len(colleges) < 2:
            continue
        
        # Iterate through each pair of colleges
        for i, college1 in enumerate(colleges):
            for college2 in colleges[i+1:]:
                # Update the matrix (symmetric)
                collaboration_matrix.at[college1, college2] += 1
                collaboration_matrix.at[college2, college1] += 1
                
                # Track the projects for this college pair
                pair_key = tuple(sorted([college1, college2]))
                if pair_key not in college_pair_projects:
                    college_pair_projects[pair_key] = []
                
                college_pair_projects[pair_key].append({
                    'grant_code': project['grant_code'],
                    'fiscal_year': project['fiscal_year'],
                    'pis': project['pis'],
                    'co_pis': project['co_pis'],
                    'college_units': project['college_units'],
                    'award_amount': project['award_amount']
                })
    
    return collaboration_matrix, college_pair_projects

def compute_college_centrality_metrics(collaboration_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute network centrality metrics for each college based on collaboration patterns.
    
    This function calculates various network centrality metrics to identify colleges that
    act as central hubs in the collaboration network.
    
    Args:
        collaboration_matrix: DataFrame containing the pairwise college collaboration counts
        
    Returns:
        DataFrame with centrality metrics for each college
    """
    try:
        import networkx as nx
    except ImportError:
        # Provide a fallback if networkx isn't available
        print("NetworkX library not found. Using simplified centrality metrics.")
        return compute_simplified_centrality_metrics(collaboration_matrix)
    
    # Create a graph from the collaboration matrix
    G = nx.from_pandas_adjacency(collaboration_matrix)
    
    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Create a DataFrame with the results
    centrality_df = pd.DataFrame({
        'college': list(degree_centrality.keys()),
        'degree_centrality': list(degree_centrality.values()),
        'betweenness_centrality': list(betweenness_centrality.values()),
        'eigenvector_centrality': list(eigenvector_centrality.values())
    })
    
    # Add total collaborations (sum of collaborative projects)
    centrality_df['total_collaborations'] = centrality_df['college'].apply(
        lambda college: collaboration_matrix.loc[college].sum()
    )
    
    # Add unique collaborators (number of colleges with at least one collaboration)
    centrality_df['unique_collaborators'] = centrality_df['college'].apply(
        lambda college: (collaboration_matrix.loc[college] > 0).sum()
    )
    
    # Sort by degree centrality in descending order
    centrality_df = centrality_df.sort_values('degree_centrality', ascending=False)
    
    return centrality_df

def compute_simplified_centrality_metrics(collaboration_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simplified centrality metrics when NetworkX is not available.
    
    This is a fallback function that computes basic centrality measures without
    requiring the NetworkX library.
    
    Args:
        collaboration_matrix: DataFrame containing the pairwise college collaboration counts
        
    Returns:
        DataFrame with simplified centrality metrics for each college
    """
    # Get all colleges
    colleges = collaboration_matrix.index.tolist()
    
    # Calculate simplified metrics
    metrics = []
    for college in colleges:
        # Total collaborations (sum of row)
        total_collaborations = collaboration_matrix.loc[college].sum()
        
        # Unique collaborators (number of non-zero entries)
        unique_collaborators = (collaboration_matrix.loc[college] > 0).sum()
        
        # Simplified degree centrality (normalized by max possible connections)
        simplified_degree = unique_collaborators / (len(colleges) - 1) if len(colleges) > 1 else 0
        
        metrics.append({
            'college': college,
            'total_collaborations': total_collaborations,
            'unique_collaborators': unique_collaborators,
            'degree_centrality': simplified_degree,
            # Placeholder values for metrics we can't compute without networkx
            'betweenness_centrality': 0,
            'eigenvector_centrality': 0
        })
    
    # Convert to DataFrame and sort
    centrality_df = pd.DataFrame(metrics)
    centrality_df = centrality_df.sort_values('total_collaborations', ascending=False)
    
    return centrality_df

def analyze_college_role_distribution(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze the distribution of PI vs. Co-PI roles for each college across projects.
    
    This function determines whether colleges predominantly lead projects (PI role) or
    contribute as partners (Co-PI role).
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        DataFrame with role distribution metrics for each college
    """
    if 'AwardsRawData' not in dataframes:
        print("Warning: AwardsRawData sheet not found")
        return pd.DataFrame()
    
    df = dataframes['AwardsRawData']
    
    required_cols = ['collegeunit', 'pi', 'co_pi_list', 'grant_code', 'fiscal_year']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Create a person-to-college mapping
    person_to_college = {}
    for _, row in df.iterrows():
        if not pd.isna(row['collegeunit']) and not pd.isna(row['pi']):
            person_to_college[row['pi']] = row['collegeunit']
    
    # Check if AwardsCoPIsRawData is available for more accurate Co-PI mapping
    if 'AwardsCoPIsRawData' in dataframes:
        copi_df = dataframes['AwardsCoPIsRawData']
        if 'co_pi' in copi_df.columns and 'collegeunit' in copi_df.columns:
            for _, row in copi_df.iterrows():
                if not pd.isna(row['co_pi']) and not pd.isna(row['collegeunit']):
                    person_to_college[row['co_pi']] = row['collegeunit']
    
    # Initialize role counters for each college
    college_roles = {}
    
    # Track unique projects for each college
    for _, row in df.iterrows():
        college = row['collegeunit']
        if pd.isna(college):
            continue
        
        # Get unique project identifier
        project_id = (row['grant_code'], row['fiscal_year'])
        
        # Initialize if this college hasn't been seen yet
        if college not in college_roles:
            college_roles[college] = {
                'pi_count': 0,  # Number of times college acts as PI
                'co_pi_count': 0,  # Number of times college acts as Co-PI
                'pi_projects': set(),  # Unique projects where college is PI
                'co_pi_projects': set(),  # Unique projects where college is Co-PI
                'total_projects': set()  # All unique projects
            }
        
        # Add to total projects
        college_roles[college]['total_projects'].add(project_id)
        
        # Check if PI is from this college
        if not pd.isna(row['pi']):
            pi_college = person_to_college.get(row['pi'])
            if pi_college == college:
                college_roles[college]['pi_count'] += 1
                college_roles[college]['pi_projects'].add(project_id)
        
        # Check if any Co-PIs are from this college
        if isinstance(row['co_pi_list'], list):
            for co_pi in row['co_pi_list']:
                if co_pi and not pd.isna(co_pi):
                    co_pi_college = person_to_college.get(co_pi)
                    if co_pi_college == college:
                        college_roles[college]['co_pi_count'] += 1
                        college_roles[college]['co_pi_projects'].add(project_id)
                        break  # Only count once per project
    
    # Convert to DataFrame
    role_data = []
    for college, roles in college_roles.items():
        unique_pi_projects = len(roles['pi_projects'])
        unique_co_pi_projects = len(roles['co_pi_projects'])
        total_unique_projects = len(roles['total_projects'])
        
        # Calculate percentages
        pi_percentage = (unique_pi_projects / total_unique_projects * 100) if total_unique_projects > 0 else 0
        co_pi_percentage = (unique_co_pi_projects / total_unique_projects * 100) if total_unique_projects > 0 else 0
        
        # Calculate leadership ratio (PI projects / Co-PI projects)
        leadership_ratio = (unique_pi_projects / unique_co_pi_projects) if unique_co_pi_projects > 0 else float('inf')
        
        role_data.append({
            'college': college,
            'pi_projects': unique_pi_projects,
            'co_pi_projects': unique_co_pi_projects,
            'total_projects': total_unique_projects,
            'pi_percentage': pi_percentage,
            'co_pi_percentage': co_pi_percentage,
            'leadership_ratio': leadership_ratio
        })
    
    # Convert to DataFrame and sort by total projects
    return pd.DataFrame(role_data).sort_values('total_projects', ascending=False)

def get_college_collaboration_diversity(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Measure how diverse each college's collaborations are.
    
    This function calculates diversity metrics to determine whether a college
    collaborates broadly with many partners or concentrates on a few partners.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        
    Returns:
        DataFrame with collaboration diversity metrics for each college
    """
    # Get college collaboration metrics
    college_metrics = get_college_collaboration_metrics(dataframes)
    
    if college_metrics.empty:
        return pd.DataFrame()
    
    # Calculate collaboration matrix to get pairwise collaboration counts
    collaboration_matrix, _ = analyze_college_collaboration_network(dataframes)
    
    # Calculate diversity metrics for each college
    diversity_metrics = []
    
    for _, row in college_metrics.iterrows():
        college = row['college']
        
        # Skip if no collaborations
        if row['collaboration_partner_count'] == 0:
            continue
        
        # Get collaboration counts with each partner
        if college in collaboration_matrix.index:
            partner_counts = collaboration_matrix.loc[college]
            # Filter to only include actual partners (non-zero counts)
            partner_counts = partner_counts[partner_counts > 0]
            
            # Calculate diversity metrics
            total_collaborations = partner_counts.sum()
            unique_partners = len(partner_counts)
            
            # Calculate Herfindahl-Hirschman Index (HHI) - a measure of concentration
            # Higher HHI means more concentrated (less diverse) collaborations
            if total_collaborations > 0:
                partner_shares = partner_counts / total_collaborations
                hhi = (partner_shares ** 2).sum()
            else:
                hhi = 1.0  # Maximum concentration if no collaborations
            
            # Calculate diversity index (1 - HHI)
            # Higher value means more diverse collaborations
            diversity_index = 1 - hhi
            
            # Calculate effective number of partners (1/HHI)
            effective_partners = 1 / hhi if hhi > 0 else 0
            
            diversity_metrics.append({
                'college': college,
                'total_collaborations': total_collaborations,
                'unique_partners': unique_partners,
                'collaboration_concentration': hhi,
                'collaboration_diversity': diversity_index,
                'effective_partners': effective_partners,
                'partners': row['collaboration_partners']
            })
    
    # Convert to DataFrame
    diversity_df = pd.DataFrame(diversity_metrics)
    
    # Sort by diversity index in descending order (most diverse first)
    if not diversity_df.empty:
        diversity_df = diversity_df.sort_values('collaboration_diversity', ascending=False)
    
    return diversity_df

def get_college_collaboration_projects(dataframes: Dict[str, pd.DataFrame], college1: str, college2: str) -> pd.DataFrame:
    """
    Get detailed project information for collaborations between two specific colleges.
    
    Args:
        dataframes: Dictionary mapping sheet names to DataFrames
        college1: First college name
        college2: Second college name
        
    Returns:
        DataFrame with detailed project information for the college pair
    """
    # Get project data
    project_data = identify_multi_college_projects(dataframes)
    
    if project_data.empty:
        return pd.DataFrame()
    
    # Filter to projects involving both colleges
    filtered_projects = []
    
    for _, project in project_data.iterrows():
        colleges = set(project['college_units'])
        if college1 in colleges and college2 in colleges:
            filtered_projects.append({
                'grant_code': project['grant_code'],
                'fiscal_year': project['fiscal_year'],
                'pis': project['pis'],
                'co_pis': project['co_pis'],
                'college_units': project['college_units'],
                'college_count': project['college_count'],
                'award_amount': project['award_amount']
            })
    
    if not filtered_projects:
        return pd.DataFrame()
    
    # Convert to DataFrame
    return pd.DataFrame(filtered_projects) 