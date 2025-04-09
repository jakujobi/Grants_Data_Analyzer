"""
UI components module for the Grants Data Analyzer application.

This module contains reusable UI components that can be used across the application.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging

# Configure logging
logger = logging.getLogger(__name__)

def detailed_data_view(
    data: pd.DataFrame,
    title: str = "Detailed Data View",
    key_columns: List[str] = None,
    formatter: Dict[str, Callable] = None,
    page_size: int = 10,
    allow_export: bool = True,
    unique_key: str = "detailed_view"
) -> None:
    """
    Display a detailed, interactive data view with expandable rows, pagination, and optional export.
    
    This component provides a rich data exploration interface with the following features:
    - Expandable rows to view all details of each record
    - Pagination controls for large datasets
    - Column filtering and sorting
    - CSV/Excel export options
    - Custom formatting for specific columns
    
    Args:
        data: DataFrame containing the data to display
        title: Title to display above the data view
        key_columns: List of column names to use as identifiers and display in the expander headers
        formatter: Dictionary mapping column names to formatting functions
        page_size: Number of rows to display per page
        allow_export: Whether to show export options
        unique_key: Unique key for this component (needed when using multiple instances on the same page)
        
    Returns:
        None
    """
    if data.empty:
        st.info(f"No data available for {title}")
        return
    
    # Initialize session state for pagination if not exists
    if f"{unique_key}_page" not in st.session_state:
        st.session_state[f"{unique_key}_page"] = 0
    
    # Display title
    st.markdown(f"### {title}")
    
    # Basic statistics
    st.markdown(f"**Total records:** {len(data)}")
    
    # Create columns for controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search functionality
        search_term = st.text_input(
            "Search", 
            key=f"{unique_key}_search",
            placeholder="Enter search term...",
            help="Search across all text columns"
        )
    
    with col2:
        # Display options
        display_mode = st.selectbox(
            "Display Mode",
            options=["Tabular", "Cards", "Expandable"],
            index=2,
            key=f"{unique_key}_display_mode"
        )
    
    # Apply search filter if search term is provided
    filtered_data = data
    if search_term:
        # Apply search to string columns
        string_cols = data.select_dtypes(include=['object']).columns
        search_mask = pd.Series(False, index=data.index)
        
        for col in string_cols:
            # Skip list columns
            if isinstance(data[col].iloc[0], list) if not data.empty else False:
                continue
            search_mask |= data[col].astype(str).str.contains(search_term, case=False, na=False)
        
        filtered_data = data[search_mask]
        st.markdown(f"**Showing {len(filtered_data)} of {len(data)} records matching '{search_term}'**")
    
    # Calculate pagination
    total_pages = max(1, (len(filtered_data) - 1) // page_size + 1)
    current_page = min(st.session_state[f"{unique_key}_page"], total_pages - 1)
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("← Previous", key=f"{unique_key}_prev", disabled=(current_page == 0)):
            st.session_state[f"{unique_key}_page"] = max(0, current_page - 1)
            st.rerun()
    
    with col2:
        st.markdown(f"**Page {current_page + 1} of {total_pages}**", 
                   help=f"Showing records {current_page * page_size + 1} to {min((current_page + 1) * page_size, len(filtered_data))} of {len(filtered_data)}")
    
    with col3:
        if st.button("Next →", key=f"{unique_key}_next", disabled=(current_page >= total_pages - 1)):
            st.session_state[f"{unique_key}_page"] = min(total_pages - 1, current_page + 1)
            st.rerun()
    
    # Get data for current page
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, len(filtered_data))
    page_data = filtered_data.iloc[start_idx:end_idx].copy()
    
    # Display data based on the selected mode
    if display_mode == "Tabular":
        st.dataframe(
            page_data,
            use_container_width=True,
            hide_index=True
        )
    
    elif display_mode == "Cards":
        # Display data as cards
        for idx, row in page_data.iterrows():
            with st.container():
                st.markdown("---")
                cols = st.columns([2, 3])
                
                # Determine key for this record
                if key_columns:
                    key_values = [f"{row[col]}" for col in key_columns if col in row]
                    record_key = " - ".join(key_values)
                else:
                    record_key = f"Record {idx}"
                
                # First column: Key information
                with cols[0]:
                    st.markdown(f"**{record_key}**")
                
                # Second column: Other fields
                with cols[1]:
                    for col in row.index:
                        if key_columns and col in key_columns:
                            continue
                        
                        value = row[col]
                        # Apply formatter if available
                        if formatter and col in formatter:
                            try:
                                value = formatter[col](value)
                            except Exception as e:
                                logger.warning(f"Error formatting column {col}: {e}")
                        
                        # Handle list values
                        if isinstance(value, list):
                            if value:
                                st.markdown(f"**{col}:**")
                                for item in value:
                                    st.markdown(f"- {item}")
                            else:
                                st.markdown(f"**{col}:** *None*")
                        else:
                            st.markdown(f"**{col}:** {value}")
    
    else:  # Expandable
        # Display data as expandable sections
        for idx, row in page_data.iterrows():
            # Determine key for this record
            if key_columns:
                key_values = [f"{row[col]}" for col in key_columns if col in row]
                record_key = " - ".join(key_values)
            else:
                record_key = f"Record {idx}"
            
            with st.expander(record_key):
                # Create two columns
                cols = st.columns([1, 1])
                
                # Divide columns into two groups
                half_point = len(row) // 2
                
                for i, (col_name, value) in enumerate(row.items()):
                    # Select column based on position
                    col_idx = 0 if i < half_point else 1
                    
                    with cols[col_idx]:
                        # Apply formatter if available
                        if formatter and col_name in formatter:
                            try:
                                formatted_value = formatter[col_name](value)
                            except Exception as e:
                                logger.warning(f"Error formatting column {col_name}: {e}")
                                formatted_value = value
                        else:
                            formatted_value = value
                        
                        # Handle list values
                        if isinstance(value, list):
                            if value:
                                st.markdown(f"**{col_name}:**")
                                for item in value:
                                    st.markdown(f"- {item}")
                            else:
                                st.markdown(f"**{col_name}:** *None*")
                        else:
                            st.markdown(f"**{col_name}:** {formatted_value}")
    
    # Export options
    if allow_export:
        st.markdown("---")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export Current Page to CSV", key=f"{unique_key}_export_csv"):
                csv = page_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{title.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"{unique_key}_download_csv"
                )
        
        with export_col2:
            if st.button("Export All Results to CSV", key=f"{unique_key}_export_all_csv"):
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{title.lower().replace(' ', '_')}_all.csv",
                    mime="text/csv",
                    key=f"{unique_key}_download_all_csv"
                )

def project_details_view(
    projects: pd.DataFrame,
    title: str = "Project Details",
    unique_key: str = "project_details"
) -> None:
    """
    Display a specialized view for project details with Co-PI information.
    
    This component builds on the detailed_data_view but is specialized for
    displaying project information including PIs, Co-PIs, colleges, etc.
    
    Args:
        projects: DataFrame containing project data with PI, Co-PI lists, etc.
        title: Title to display above the data view
        unique_key: Unique key for this component (needed when using multiple instances on the same page)
        
    Returns:
        None
    """
    if projects.empty:
        st.info(f"No projects available for {title}")
        return
    
    # Define custom formatters
    formatters = {
        'award_amount': lambda x: f"${x:,.2f}" if x and not pd.isna(x) else "N/A"
    }
    
    # Identify key columns for the expander headers
    key_columns = ['grant_code', 'fiscal_year']
    
    # Call the generic detailed data view with project-specific configurations
    detailed_data_view(
        data=projects,
        title=title,
        key_columns=key_columns,
        formatter=formatters,
        page_size=5,  # Projects often have more detailed information, so show fewer per page
        allow_export=True,
        unique_key=unique_key
    )

def create_expandable_project_list(
    projects: pd.DataFrame,
    fiscal_year: Optional[int] = None,
    title: str = "Projects",
    show_metrics: bool = True,
    unique_key: str = "project_list"
) -> None:
    """
    Create an expandable list of projects with detailed information for each project.
    
    This component shows projects in an interactive list with expandable details for each project.
    It provides summary metrics and can be filtered by fiscal year.
    
    Args:
        projects: DataFrame containing project data (raw project data with lists for co_pis, pis, etc.)
        fiscal_year: Optional fiscal year to filter by
        title: Title to display above the project list
        show_metrics: Whether to show summary metrics above the list
        unique_key: Unique key for this component (needed when using multiple instances on the same page)
        
    Returns:
        None
    """
    if projects.empty:
        st.info(f"No projects available for {title}")
        return
    
    # Filter by fiscal year if provided
    if fiscal_year is not None:
        filtered_projects = projects[projects['fiscal_year'] == fiscal_year]
    else:
        filtered_projects = projects
    
    if filtered_projects.empty:
        st.info(f"No projects available for fiscal year {fiscal_year}")
        return
    
    # Display title
    st.markdown(f"### {title}")
    
    # Show summary metrics if requested
    if show_metrics:
        # Create summary metrics
        total_projects = len(filtered_projects)
        total_pis = sum(len(project['pis']) if isinstance(project['pis'], list) else 0 
                        for _, project in filtered_projects.iterrows())
        total_co_pis = sum(len(project['co_pis']) if isinstance(project['co_pis'], list) else 0 
                          for _, project in filtered_projects.iterrows())
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Projects", f"{total_projects:,}")
        with col2:
            st.metric("Total PIs", f"{total_pis:,}")
        with col3:
            st.metric("Total Co-PIs", f"{total_co_pis:,}")
    
    # Create expandable sections for each project
    for idx, project in filtered_projects.iterrows():
        # Create expander with grant code and fiscal year
        with st.expander(f"{project['grant_code']} (FY {int(project['fiscal_year'])})"):
            # Create columns for project details
            col1, col2 = st.columns([1, 1])
            
            # First column: Basic project info
            with col1:
                st.markdown("**Project Information**")
                st.markdown(f"**Grant Code:** {project['grant_code']}")
                st.markdown(f"**Fiscal Year:** {int(project['fiscal_year'])}")
                
                # Award amount if available
                if 'award_amount' in project and project['award_amount'] is not None and not pd.isna(project['award_amount']):
                    st.markdown(f"**Award Amount:** ${project['award_amount']:,.2f}")
                
                # Display Principal Investigators
                st.markdown("**Principal Investigators:**")
                if isinstance(project['pis'], list) and project['pis']:
                    for pi in project['pis']:
                        st.markdown(f"- {pi}")
                else:
                    st.markdown("*None listed*")
            
            # Second column: Co-PIs and Colleges
            with col2:
                # Display Co-Principal Investigators
                st.markdown("**Co-Principal Investigators:**")
                if isinstance(project['co_pis'], list) and project['co_pis']:
                    for co_pi in project['co_pis']:
                        st.markdown(f"- {co_pi}")
                else:
                    st.markdown("*None listed*")
                
                # Display College Units
                st.markdown("**College Units:**")
                if isinstance(project['college_units'], list) and project['college_units']:
                    for unit in project['college_units']:
                        st.markdown(f"- {unit}")
                else:
                    st.markdown("*None listed*") 