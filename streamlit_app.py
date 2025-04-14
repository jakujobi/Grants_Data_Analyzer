import streamlit as st
import pandas as pd
import os
import tempfile
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import logging
import sys
import re
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def add_debug_info(message: str):
    """Add debug information to the session state and log it."""
    logger.debug(message)
    if 'debug_info' in st.session_state:
        st.session_state.debug_info.append(message)

# Import modules
from modules.load_data import load_excel_file, validate_dataframes, get_fiscal_years, get_college_units, normalize_column_names
from modules.clean_data import clean_data
from modules.analysis import (
    count_multi_college_projects, count_multi_co_pi_projects,
    get_colleges_with_most_collaborators, get_project_counts_by_person,
    get_project_counts_by_college, get_summary_counts,
    filter_by_fiscal_year, filter_by_fiscal_year_range,
    get_projects_by_co_pi_count, get_projects_by_co_pi_count_yearly,
    get_project_details_by_co_pi_filter,
    identify_multi_college_projects, get_multi_college_project_yearly_stats,
    get_college_collaboration_metrics, get_multi_college_vs_single_college_comparison,
    get_multi_college_projects_by_year,
    create_college_collaboration_network, get_college_role_distribution,
    get_college_collaboration_diversity, get_pairwise_college_collaborations
)
from modules.visuals import (
    create_bar_chart, create_pie_chart, create_line_chart,
    create_summary_dashboard, save_figure,
    create_co_pi_analysis_chart, create_co_pi_comparison_chart,
    create_multi_college_trend_chart, create_multi_vs_single_college_comparison_chart,
    create_college_collaboration_chart, create_avg_colleges_per_project_chart,
    create_multi_college_dashboard,
    create_network_graph, create_heatmap, create_role_distribution_chart,
    create_collaboration_diversity_chart, create_college_collaboration_dashboard
)
from modules.utils import export_to_excel, export_to_csv, ensure_directory_exists
from modules.components import detailed_data_view, project_details_view, create_expandable_project_list

# Set page config
st.set_page_config(
    page_title="Grant Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info("Application started")

# SDSU brand colors
SDSU_BLUE = "#0033A0"
SDSU_YELLOW = "#FFD100"
SDSU_WHITE = "#FFFFFF"

# Security and configuration settings
DEFAULT_MAX_UPLOAD_SIZE_MB = 200
ALLOWED_FILE_TYPES = ["xlsx", "xls"]
MAX_PROCESSING_TIME_SECONDS = 300  # 5 minutes timeout for processing

# Custom CSS
st.markdown(f"""
<style>
    .main-header {{
        color: {SDSU_BLUE};
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        color: {SDSU_BLUE};
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    .stButton>button {{
        background-color: {SDSU_BLUE};
        color: {SDSU_WHITE};
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {SDSU_YELLOW};
        color: {SDSU_BLUE};
    }}
    .info-box {{
        background-color: {SDSU_WHITE};
        border: 1px solid {SDSU_BLUE};
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    .debug-info {{
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
    }}
    .warning-box {{
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    .error-box {{
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">SDSU Grant Data Explorer</h1>', unsafe_allow_html=True)

# Debug information expander
with st.expander("Debug Information"):
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = []
    
    for info in st.session_state.debug_info:
        st.markdown(f'<div class="debug-info">{info}</div>', unsafe_allow_html=True)

# Initialize session state
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

if 'filtered_dataframes' not in st.session_state:
    st.session_state.filtered_dataframes = {}

if 'college_units' not in st.session_state:
    st.session_state.college_units = []

if 'fiscal_years' not in st.session_state:
    st.session_state.fiscal_years = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

if 'max_upload_size_mb' not in st.session_state:
    st.session_state.max_upload_size_mb = DEFAULT_MAX_UPLOAD_SIZE_MB

if 'processing_start_time' not in st.session_state:
    st.session_state.processing_start_time = None

# Sidebar
with st.sidebar:
    st.markdown(f'<h2 style="color: {SDSU_BLUE};">Upload Data</h2>', unsafe_allow_html=True)
    
    # Configuration section
    with st.expander("Configuration"):
        st.markdown("### Security Settings")
        max_upload_size = st.number_input(
            "Maximum file upload size (MB)", 
            min_value=1, 
            max_value=200, 
            value=st.session_state.max_upload_size_mb,
            help="Set the maximum allowed file size for uploads"
        )
        st.session_state.max_upload_size_mb = max_upload_size
        
        st.markdown("### Advanced Settings")
        show_debug_info = st.checkbox("Show debug information", value=True)
    
    # File uploader with size limit
    uploaded_file = st.file_uploader(
        f"Upload Excel file (max {st.session_state.max_upload_size_mb}MB)", 
        type=ALLOWED_FILE_TYPES
    )
    
    # Sheet selection
    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > st.session_state.max_upload_size_mb:
            st.error(f"File size ({file_size_mb:.2f}MB) exceeds the maximum allowed size ({st.session_state.max_upload_size_mb}MB)")
            uploaded_file = None
        else:
            add_debug_info(f"File uploaded: {uploaded_file.name} (Size: {file_size_mb:.2f}MB)")
            
            # Save the uploaded file to a temporary file
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                    add_debug_info(f"Temporary file created: {tmp_file_path}")
                
                # Load the Excel file to get sheet names
                try:
                    xl = pd.ExcelFile(tmp_file_path)
                    sheet_names = xl.sheet_names
                    add_debug_info(f"Available sheets: {sheet_names}")
                    
                    # Default sheet selection
                    default_sheets = ["AwardsRawData", "AwardsCoPIsRawData"]
                    selected_sheets = st.multiselect(
                        "Select sheets to analyze",
                        sheet_names,
                        default=[sheet for sheet in default_sheets if sheet in sheet_names]
                    )
                    add_debug_info(f"Selected sheets: {selected_sheets}")
                except Exception as e:
                    error_msg = f"Error loading Excel file: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    selected_sheets = []
            except Exception as e:
                error_msg = f"Error saving uploaded file: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                selected_sheets = []
    else:
        selected_sheets = []
        tmp_file_path = None

    # Process the uploaded file and update session state if not already done
    if (uploaded_file is not None and selected_sheets and not st.session_state.data_loaded):
        try:
            # Set processing start time
            st.session_state.processing_start_time = time.time()
            
            # Show processing indicator
            with st.spinner("Processing data... This may take a moment."):
                dataframes = load_excel_file(tmp_file_path, selected_sheets)
                add_debug_info("Data loaded from Excel file")
                
                for sheet_name, df in dataframes.items():
                    add_debug_info(f"\nSheet: {sheet_name}")
                    add_debug_info(f"Shape: {df.shape}")
                    add_debug_info(f"Columns: {list(df.columns)}")
                    add_debug_info(f"Missing values:\n{df.isnull().sum()}")
                
                # Validate data
                is_valid, errors = validate_dataframes(dataframes)
                if not is_valid:
                    st.error("Data validation failed:")
                    for error in errors:
                        st.error(error)
                    st.stop()
                
                add_debug_info("\nData validation passed")
                
                # Normalize column names
                normalized_dataframes = normalize_column_names(dataframes)
                add_debug_info("\nColumn names normalized")
                
                for sheet_name, df in normalized_dataframes.items():
                    add_debug_info(f"\nNormalized sheet: {sheet_name}")
                    add_debug_info(f"Columns after normalization: {list(df.columns)}")
                
                # Clean data
                cleaned_dataframes = clean_data(normalized_dataframes)
                
                for sheet_name, df in cleaned_dataframes.items():
                    add_debug_info(f"\nCleaned sheet: {sheet_name}")
                    add_debug_info(f"Shape after cleaning: {df.shape}")
                    add_debug_info(f"Columns after cleaning: {list(df.columns)}")
                    add_debug_info(f"Missing values after cleaning:\n{df.isnull().sum()}")
                
                # Store in session state
                st.session_state.dataframes = cleaned_dataframes.copy()
                st.session_state.filtered_dataframes = cleaned_dataframes.copy()
                
                # Get fiscal years and college units
                fiscal_years = get_fiscal_years(cleaned_dataframes)
                st.session_state.fiscal_years = sorted(fiscal_years) if fiscal_years else []
                st.session_state.college_units = get_college_units(cleaned_dataframes)
                
                add_debug_info(f"\nFiscal years found: {st.session_state.fiscal_years}")
                add_debug_info(f"College units found: {st.session_state.college_units}")
                
                st.session_state.data_loaded = True
                
                # Calculate processing time
                processing_time = time.time() - st.session_state.processing_start_time
                add_debug_info(f"Processing completed in {processing_time:.2f} seconds")
                
                st.success("Data loaded successfully!")
                
                # Clean up temporary file
                try:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        add_debug_info(f"Temporary file removed: {tmp_file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file: {str(e)}")
                
        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.stop()
    
    # Fiscal year update
    if st.session_state.get('dataframes'):
        computed_years = get_fiscal_years(st.session_state.dataframes)
        st.session_state.fiscal_years = sorted(computed_years) if computed_years else []

    # Filter options
    st.markdown(f'<h2 style="color: {SDSU_BLUE};">Filter Options</h2>', unsafe_allow_html=True)
    
    # Fiscal year filter
    fiscal_year_filter = st.radio(
        "Fiscal Year Filter",
        ["All Years", "Single Year", "Year Range"],
        key='fiscal_year_filter'
    )
    add_debug_info(f"Selected fiscal year filter: {fiscal_year_filter}")

    # Show year selection controls based on filter type
    if len(st.session_state.fiscal_years) == 0:
        st.info("Upload data to enable year filtering")
        add_debug_info("No fiscal years available in session state")
    else:
        add_debug_info(f"Available fiscal years: {st.session_state.fiscal_years}")
        
        if fiscal_year_filter == "Single Year":
            selected_year = st.selectbox(
                "Select Fiscal Year",
                options=sorted(st.session_state.fiscal_years),
                key='single_year_select'
            )
            
            if st.button("Apply Single Year Filter", key='apply_single_year'):
                try:
                    st.session_state.filtered_dataframes = filter_by_fiscal_year(
                        st.session_state.dataframes, selected_year
                    )
                    st.success(f"Filtered data for fiscal year {selected_year}")
                except Exception as e:
                    st.error(f"Error filtering data: {str(e)}")
        
        elif fiscal_year_filter == "Year Range":
            col1, col2 = st.columns(2)
            
            with col1:
                start_year = st.selectbox(
                    "Start Year",
                    options=sorted(st.session_state.fiscal_years),
                    index=0,
                    key='start_year_select'
                )
            
            with col2:
                valid_end_years = [year for year in st.session_state.fiscal_years if year >= start_year]
                end_year = st.selectbox(
                    "End Year",
                    options=valid_end_years,
                    index=len(valid_end_years)-1 if valid_end_years else 0,
                    key='end_year_select'
                )
            
            if st.button("Apply Year Range Filter", key='apply_year_range'):
                try:
                    st.session_state.filtered_dataframes = filter_by_fiscal_year_range(
                        st.session_state.dataframes, start_year, end_year
                    )
                    st.success(f"Filtered data for fiscal years {start_year}-{end_year}")
                except Exception as e:
                    st.error(f"Error filtering data: {str(e)}")
        
        elif fiscal_year_filter == "All Years":
            if st.button("Reset to All Years", key='reset_years'):
                st.session_state.filtered_dataframes = st.session_state.dataframes.copy()
                st.success("Reset to all years")

# Main content
if st.session_state.dataframes:
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Summary", "Collaborators", 
        "Project Counts", "College Analysis", "Projects with X Co-PIs", 
        "Multi-College Projects Analyzer", "College Collaboration Network", "Export Data"
    ])
    
    # Summary tab
    with tab1:
        st.markdown(f'<h2 class="sub-header">Summary Statistics</h2>', unsafe_allow_html=True)
        
        # Get summary counts
        summary_counts = get_summary_counts(st.session_state.filtered_dataframes)
        
        # Display summary counts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Unique PIs", summary_counts['unique_pis'])
        
        with col2:
            st.metric("Unique Co-PIs", summary_counts['unique_co_pis'])
        
        with col3:
            st.metric("Unique Colleges", summary_counts['unique_colleges'])
        
        # Create summary dashboard
        fig = create_summary_dashboard(summary_counts)
        st.pyplot(fig)
    
    # Collaborators tab
    with tab2:
        st.markdown(f'<h2 class="sub-header">Colleges with Most Collaborators</h2>', unsafe_allow_html=True)
        
        # Get colleges with most collaborators
        top_n = st.slider("Number of top colleges to show", 5, 20, 10)
        colleges_data = get_colleges_with_most_collaborators(
            st.session_state.filtered_dataframes, top_n
        )
        
        # Display data
        st.dataframe(colleges_data)
        
        # Create visualization
        fig = create_bar_chart(
            colleges_data, 'collegeunit', 'collaborator_count',
            'Top Colleges by Number of Collaborators',
            x_label='College',
            y_label='Number of Collaborators'
        )
        st.pyplot(fig)
    
    # Project Counts tab
    with tab3:
        st.markdown(f'<h2 class="sub-header">Project Counts by Person</h2>', unsafe_allow_html=True)
        
        # Get project counts by person
        person_data = get_project_counts_by_person(st.session_state.filtered_dataframes)
        
        # Filter options
        role_filter = st.radio(
            "Filter by Role",
            ["All", "PI Only", "Co-PI Only"]
        )
        
        if role_filter == "PI Only":
            filtered_data = person_data[person_data['role'] == 'PI']
        elif role_filter == "Co-PI Only":
            filtered_data = person_data[person_data['role'] == 'Co-PI']
        else:
            filtered_data = person_data
        
        # Display top N people
        top_n = st.slider("Number of top people to show", 5, 50, 20)
        top_people = filtered_data.head(top_n)
        
        # Display data
        st.dataframe(top_people)
        
        # Create visualization
        fig = create_bar_chart(
            top_people, 'person', 'project_count',
            f'Top {top_n} {role_filter} by Project Count',
            x_label='Person',
            y_label='Number of Projects'
        )
        st.pyplot(fig)
    
    # College Analysis tab
    with tab4:
        st.markdown(f'<h2 class="sub-header">Projects per College</h2>', unsafe_allow_html=True)
        
        # Get project counts by college
        college_data = get_project_counts_by_college(st.session_state.filtered_dataframes)
        
        # Display data
        st.dataframe(college_data)
        
        # Create visualization
        fig = create_pie_chart(
            college_data, 'project_count', 'collegeunit',
            'Project Distribution by College',
            top_n=5
        )
        st.pyplot(fig)
    
    # Projects with X Co-PIs tab
    with tab5:
        st.markdown(f'<h2 class="sub-header">Projects with X Co-PIs Analysis</h2>', unsafe_allow_html=True)
        
        # Create columns for filter controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Co-PI filter condition
            condition_options = {
                "at_least": "At Least",
                "at_most": "At Most", 
                "exactly": "Exactly",
                "more_than": "More Than", 
                "less_than": "Less Than"
            }
            condition = st.selectbox(
                "Co-PI Count Condition",
                options=list(condition_options.keys()),
                format_func=lambda x: condition_options[x],
                index=0
            )
        
        with filter_col2:
            # Co-PI count
            co_pi_count = st.number_input(
                "Number of Co-PIs",
                min_value=0,
                max_value=20,
                value=0
            )
        
        with filter_col3:
            # Visualization metric
            metric_options = {
                "matching_projects": "Matching Projects Count",
                "matching_percentage": "Matching Projects Percentage",
                "total_co_pis": "Total Co-PIs Count"
            }
            visualization_metric = st.selectbox(
                "Visualization Metric",
                options=list(metric_options.keys()),
                format_func=lambda x: metric_options[x],
                index=0
            )
        
        # Analysis type
        analysis_type = st.radio(
            "Analysis Type",
            ["Yearly Trend", "Year-by-Year Details", "Detailed Project View"],
            horizontal=True,
            index=0
        )
        
        # Yearly Trend analysis
        if analysis_type == "Yearly Trend":
            # Get min and max years
            min_year = min(st.session_state.fiscal_years) if st.session_state.fiscal_years else 2010
            max_year = max(st.session_state.fiscal_years) if st.session_state.fiscal_years else 2023
            
            year_range = st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            # Filter data by year range
            year_filtered_data = filter_by_fiscal_year_range(
                st.session_state.dataframes, 
                year_range[0], 
                year_range[1]
            )
            
            # Get yearly statistics
            yearly_data = get_projects_by_co_pi_count_yearly(
                year_filtered_data, 
                condition, 
                co_pi_count
            )
            
            # Display results
            st.markdown(f"### Projects {condition_options[condition].lower()} {co_pi_count} Co-PI(s)")
            
            # Summary metrics
            if not yearly_data.empty:
                # Only show info if we have data
                total_projects = yearly_data['total_projects'].sum()
                matching_projects = yearly_data['matching_projects'].sum()
                percentage = (matching_projects / total_projects * 100) if total_projects > 0 else 0
                
                # Create metric columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Total Projects", f"{total_projects:,}")
                    
                with metric_col2:
                    st.metric("Matching Projects", f"{matching_projects:,}")
                    
                with metric_col3:
                    st.metric("Percentage", f"{percentage:.1f}%")
                
                # Visualizations
                viz_tabs = st.tabs(["Bar Chart", "Comparison Chart"])
                
                with viz_tabs[0]:
                    fig = create_co_pi_analysis_chart(yearly_data, visualization_metric)
                    st.pyplot(fig)
                
                with viz_tabs[1]:
                    fig = create_co_pi_comparison_chart(yearly_data)
                    st.pyplot(fig)
                
                # Show data table
                with st.expander("View Data Table"):
                    st.dataframe(yearly_data)
            else:
                st.info("No data available for the selected criteria.")
        
        # Year-by-Year detailed analysis
        elif analysis_type == "Year-by-Year Details":
            # Select specific year
            selected_year = st.selectbox(
                "Select Fiscal Year",
                options=sorted(st.session_state.fiscal_years),
                index=0 if st.session_state.fiscal_years else None
            )
            
            if selected_year:
                # Get detailed project data
                detailed_data = get_project_details_by_co_pi_filter(
                    st.session_state.dataframes,
                    selected_year,
                    condition,
                    co_pi_count
                )
                
                if not detailed_data.empty:
                    # Show summary
                    st.markdown(f"### Projects for Fiscal Year {selected_year}")
                    st.markdown(f"Found **{len(detailed_data)}** projects {condition_options[condition].lower()} {co_pi_count} Co-PI(s)")
                    
                    # Display the data table with column configuration
                    st.dataframe(
                        detailed_data,
                        use_container_width=True,
                        column_config={
                            "grant_code": st.column_config.TextColumn("Grant Code"),
                            "fiscal_year": st.column_config.NumberColumn("Fiscal Year"),
                            "co_pi_count": st.column_config.NumberColumn("Co-PI Count"),
                            "award_amount": st.column_config.NumberColumn("Award Amount", format="$%.2f")
                        }
                    )
                    
                    # Get raw project details for exploration
                    projects = get_projects_by_co_pi_count(
                        filter_by_fiscal_year(st.session_state.dataframes, selected_year),
                        condition,
                        co_pi_count
                    )
                    
                    # Show detailed project information in expandable sections
                    if not projects.empty:
                        st.markdown("### Project Details")
                        for i, project in projects.iterrows():
                            with st.expander(f"Project: {project['grant_code']} (FY {project['fiscal_year']})"):
                                # Project overview
                                st.markdown(f"**Grant Code:** {project['grant_code']}")
                                st.markdown(f"**Fiscal Year:** {project['fiscal_year']}")
                                st.markdown(f"**Co-PI Count:** {project['co_pi_count']}")
                                
                                if project['award_amount'] is not None:
                                    st.markdown(f"**Award Amount:** ${project['award_amount']:,.2f}")
                                
                                # PIs
                                st.markdown("**Principal Investigators:**")
                                for pi in project['pis']:
                                    st.markdown(f"- {pi}")
                                
                                # Co-PIs
                                st.markdown("**Co-Principal Investigators:**")
                                if project['co_pis']:
                                    for co_pi in project['co_pis']:
                                        st.markdown(f"- {co_pi}")
                                else:
                                    st.markdown("*No Co-PIs*")
                                
                                # College Units
                                st.markdown("**College Units:**")
                                for unit in project['college_units']:
                                    st.markdown(f"- {unit}")
                else:
                    st.info(f"No projects found for fiscal year {selected_year} with {condition_options[condition].lower()} {co_pi_count} Co-PI(s).")
        
        # Detailed Project View using the new component
        else:  # Detailed Project View
            st.markdown("### Detailed Project Analysis")
            
            # Set up filter options
            filter_options_expander = st.expander("Filter Options", expanded=True)
            
            with filter_options_expander:
                # Year range selection
                col1, col2 = st.columns(2)
                
                with col1:
                    # Get min and max years
                    min_year = min(st.session_state.fiscal_years) if st.session_state.fiscal_years else 2010
                    max_year = max(st.session_state.fiscal_years) if st.session_state.fiscal_years else 2023
                    
                    year_filter_type = st.radio(
                        "Year Filter",
                        ["All Years", "Year Range", "Single Year"],
                        horizontal=True
                    )
                
                with col2:
                    if year_filter_type == "Year Range":
                        year_range = st.slider(
                            "Select Year Range",
                            min_value=min_year,
                            max_value=max_year,
                            value=(min_year, max_year)
                        )
                        filtered_data = filter_by_fiscal_year_range(
                            st.session_state.dataframes,
                            year_range[0],
                            year_range[1]
                        )
                    elif year_filter_type == "Single Year":
                        selected_year = st.selectbox(
                            "Select Fiscal Year",
                            options=sorted(st.session_state.fiscal_years),
                            index=0 if st.session_state.fiscal_years else None
                        )
                        filtered_data = filter_by_fiscal_year(
                            st.session_state.dataframes,
                            selected_year
                        ) if selected_year else {}
                    else:  # All Years
                        filtered_data = st.session_state.dataframes
            
            if filtered_data:
                # Get project details
                projects = get_projects_by_co_pi_count(
                    filtered_data,
                    condition,
                    co_pi_count
                )
                
                if not projects.empty:
                    # Create a descriptive title based on the filter
                    if year_filter_type == "All Years":
                        title = f"Projects {condition_options[condition].lower()} {co_pi_count} Co-PI(s) - All Years"
                    elif year_filter_type == "Year Range":
                        title = f"Projects {condition_options[condition].lower()} {co_pi_count} Co-PI(s) - {year_range[0]} to {year_range[1]}"
                    else:  # Single Year
                        title = f"Projects {condition_options[condition].lower()} {co_pi_count} Co-PI(s) - FY {selected_year}"
                    
                    # Use the new component for detailed project view
                    create_expandable_project_list(
                        projects=projects,
                        title=title,
                        show_metrics=True,
                        unique_key="co_pi_projects"
                    )
                    
                    # Add advanced data exploration
                    st.markdown("### Advanced Data Exploration")
                    st.markdown("Use the interactive data view below to search, filter, and export project data.")
                    
                    # Use the detailed data view component
                    project_details_view(
                        projects=projects,
                        title="Detailed Project Data",
                        unique_key="co_pi_details"
                    )
                else:
                    st.info(f"No projects found {condition_options[condition].lower()} {co_pi_count} Co-PI(s) for the selected time period.")
            else:
                st.info("Please select valid filter options to view project details.")
    
    # Multi-College Projects Analyzer tab
    with tab6:
        st.markdown(f'<h2 class="sub-header">Multi-College Projects Analyzer</h2>', unsafe_allow_html=True)
        
        # Description of what constitutes a multi-college project
        with st.expander("What is a Multi-College Project?", expanded=False):
            st.markdown("""
            ### Multi-College Project Definition
            
            In this analysis, a project is considered a **multi-college project** when the PI and/or Co-PIs collectively represent more than one college. For example:
            
            - **Project A:** PI from Engineering, Co-PI from Natural Sciences → *Multi-college (2 colleges)*
            - **Project B:** PI and all Co-PIs from Engineering → *Not multi-college (1 college)*
            - **Project C:** PI from Engineering, one Co-PI from Natural Sciences, another Co-PI from Engineering → *Multi-college (2 colleges)*
            - **Project D:** PI from Engineering, one Co-PI from Natural Sciences, one Co-PI from Engineering, one Co-PI from Nursing → *Multi-college (3 colleges)*
            """)
        
        # Compute multi-college project data
        with st.spinner("Analyzing multi-college projects..."):
            # Get overall comparison data
            comparison_data = get_multi_college_vs_single_college_comparison(st.session_state.filtered_dataframes)
            
            # Get yearly statistics
            yearly_stats = get_multi_college_project_yearly_stats(st.session_state.filtered_dataframes)
            
            # Get college collaboration metrics
            college_metrics = get_college_collaboration_metrics(st.session_state.filtered_dataframes)
        
        # Add debug information
        with st.expander("Debug Information", expanded=False):
            st.markdown("### Data Validation")
            
            # Check for AwardsRawData sheet
            if 'AwardsRawData' in st.session_state.filtered_dataframes:
                st.success("✅ AwardsRawData sheet found")
                
                # Check for required columns
                awards_df = st.session_state.filtered_dataframes['AwardsRawData']
                required_cols = ['grant_code', 'fiscal_year', 'pi', 'co_pi_list', 'collegeunit']
                missing_cols = [col for col in required_cols if col not in awards_df.columns]
                
                if not missing_cols:
                    st.success(f"✅ All required columns found: {', '.join(required_cols)}")
                else:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info(f"Available columns: {', '.join(awards_df.columns)}")
                
                # Show collegeunit values
                if 'collegeunit' in awards_df.columns:
                    unique_colleges = awards_df['collegeunit'].dropna().unique()
                    st.markdown(f"**Unique College Units:** {len(unique_colleges)}")
                    st.write(unique_colleges)
            else:
                st.error("❌ AwardsRawData sheet not found")
            
            # Check for AwardsCoPIsRawData sheet
            if 'AwardsCoPIsRawData' in st.session_state.filtered_dataframes:
                st.success("✅ AwardsCoPIsRawData sheet found")
                
                # Check for columns
                co_pi_df = st.session_state.filtered_dataframes['AwardsCoPIsRawData']
                st.markdown(f"**Available columns:** {', '.join(co_pi_df.columns)}")
                
                if 'collegeunit' in co_pi_df.columns:
                    unique_colleges = co_pi_df['collegeunit'].dropna().unique()
                    st.markdown(f"**Unique College Units in Co-PI data:** {len(unique_colleges)}")
                    st.write(unique_colleges)
            else:
                st.warning("⚠️ AwardsCoPIsRawData sheet not found")
            
            # Show multi-college project counts
            st.markdown("### Multi-College Project Counts")
            st.json(comparison_data)
        
        # Create tabs for different analysis views
        analysis_tabs = st.tabs([
            "Dashboard", "Yearly Trends", "College Comparisons", "Detailed Projects", "Funding Analysis"
        ])
        
        # Dashboard tab (overview)
        with analysis_tabs[0]:
            st.markdown("### Multi-College Projects Dashboard")
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Multi-College Projects", 
                    f"{comparison_data['multi_college_projects']:,}",
                    f"{comparison_data['multi_college_percentage']:.1f}% of total"
                )
            
            with col2:
                st.metric(
                    "Avg. Colleges per Project", 
                    f"{comparison_data['avg_colleges_multi']:.2f}"
                )
            
            with col3:
                if comparison_data['award_difference_percentage'] is not None:
                    label = "Funding Difference"
                    value = f"{abs(comparison_data['award_difference_percentage']):.1f}%"
                    delta = "more" if comparison_data['award_difference_percentage'] > 0 else "less"
                    st.metric(label, value, delta)
                else:
                    st.metric("Funding Data", "Not Available")
            
            # Raw project data exploration
            if st.session_state.filtered_dataframes:
                # Get raw project data for debugging/exploration
                raw_projects = identify_multi_college_projects(st.session_state.filtered_dataframes)
                
                if not raw_projects.empty:
                    multi_college_count = raw_projects['is_multi_college'].sum()
                    if multi_college_count > 0:
                        with st.expander("Raw Project Data", expanded=False):
                            st.markdown(f"### Showing {len(raw_projects)} Projects ({multi_college_count} Multi-College)")
                            
                            # College count distribution
                            col_counts = raw_projects['college_count'].value_counts().sort_index()
                            count_df = pd.DataFrame({
                                'College Count': col_counts.index,
                                'Number of Projects': col_counts.values,
                                'Percentage': (col_counts.values / len(raw_projects) * 100).round(1)
                            })
                            
                            st.markdown("#### College Count Distribution")
                            st.dataframe(count_df, use_container_width=True)
                            
                            # Multi-college project examples
                            st.markdown("#### Examples of Multi-College Projects")
                            multi_examples = raw_projects[raw_projects['is_multi_college']].head(5)
                            
                            for _, project in multi_examples.iterrows():
                                st.markdown(f"**Project: {project['grant_code']} (FY {project['fiscal_year']})**")
                                st.markdown(f"- **Colleges ({len(project['college_units'])}):** {', '.join(project['college_units'])}")
                                st.markdown(f"- **PIs:** {', '.join(project['pis'])}")
                                st.markdown(f"- **Co-PIs ({len(project['co_pis'])}):** {', '.join(project['co_pis'])}")
                                st.markdown("---")
                    else:
                        st.warning("No multi-college projects found in the filtered data.")
                        
                        # Show examples of single-college projects
                        with st.expander("Single-College Project Examples", expanded=False):
                            st.markdown("#### Examples of Single-College Projects")
                            single_examples = raw_projects[~raw_projects['is_multi_college']].head(5)
                            
                            for _, project in single_examples.iterrows():
                                st.markdown(f"**Project: {project['grant_code']} (FY {project['fiscal_year']})**")
                                st.markdown(f"- **College:** {', '.join(project['college_units'])}")
                                st.markdown(f"- **PIs:** {', '.join(project['pis'])}")
                                st.markdown(f"- **Co-PIs ({len(project['co_pis'])}):** {', '.join(project['co_pis'])}")
                                st.markdown("---")
            
            # Create comprehensive dashboard
            fig = create_multi_college_dashboard(yearly_stats, comparison_data)
            st.pyplot(fig)
            
            # Allow dashboard export
            if st.button("Export Dashboard", key="export_multi_college_dashboard"):
                output_dir = "output"
                ensure_directory_exists(output_dir)
                output_path = os.path.join(output_dir, "multi_college_dashboard.png")
                
                with st.spinner("Exporting dashboard..."):
                    save_figure(fig, output_path)
                    st.success(f"Dashboard exported to {output_path}")
                    
                    # Provide download link
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Dashboard",
                            data=file,
                            file_name="multi_college_dashboard.png",
                            mime="image/png",
                            key="download_multi_college_dashboard"
                        )
        
        # Yearly Trends tab
        with analysis_tabs[1]:
            st.markdown("### Multi-College Projects Trends")
            
            # Create trend chart
            fig = create_multi_college_trend_chart(yearly_stats)
            st.pyplot(fig)
            
            # Display data table with yearly stats
            if not yearly_stats.empty:
                with st.expander("View Yearly Data", expanded=False):
                    # Format the DataFrame for display
                    display_df = yearly_stats.copy()
                    
                    # Format percentages
                    display_df['multi_college_percentage'] = display_df['multi_college_percentage'].map('{:.1f}%'.format)
                    
                    # Rename columns for better readability
                    display_df = display_df.rename(columns={
                        'fiscal_year': 'Fiscal Year',
                        'total_projects': 'Total Projects',
                        'multi_college_projects': 'Multi-College Projects',
                        'single_college_projects': 'Single-College Projects',
                        'multi_college_percentage': 'Multi-College %',
                        'avg_colleges_per_multi_project': 'Avg. Colleges per Multi-College Project'
                    })
                    
                    # Select columns to display
                    columns_to_display = ['Fiscal Year', 'Total Projects', 'Multi-College Projects', 
                                         'Single-College Projects', 'Multi-College %', 
                                         'Avg. Colleges per Multi-College Project']
                    
                    # Display the table
                    st.dataframe(display_df[columns_to_display])
        
        # College Comparisons tab
        with analysis_tabs[2]:
            st.markdown("### College Collaboration Analysis")
            
            # Controls for the number of colleges to display
            top_n = st.slider("Number of top colleges to show", 
                             min_value=5, max_value=20, value=10, 
                             key="multi_college_top_n")
            
            # Create college collaboration chart
            fig = create_college_collaboration_chart(college_metrics, top_n)
            st.pyplot(fig)
            
            # Display college metrics table
            if not college_metrics.empty:
                with st.expander("View College Metrics", expanded=False):
                    # Format the DataFrame for display
                    display_df = college_metrics.copy()
                    
                    # Format percentages
                    display_df['multi_college_percentage'] = display_df['multi_college_percentage'].map('{:.1f}%'.format)
                    
                    # Rename columns for better readability
                    display_df = display_df.rename(columns={
                        'college': 'College',
                        'total_projects': 'Total Projects',
                        'multi_college_projects': 'Multi-College Projects',
                        'multi_college_percentage': 'Multi-College %',
                        'collaboration_partner_count': 'Number of Partner Colleges',
                        'collaboration_partners': 'Partner Colleges',
                        'pi_count': 'PI Count',
                        'co_pi_count': 'Co-PI Count'
                    })
                    
                    # Select columns to display
                    columns_to_display = ['College', 'Total Projects', 'Multi-College Projects', 
                                         'Multi-College %', 'Number of Partner Colleges', 
                                         'Partner Colleges']
                    
                    # Display the table
                    st.dataframe(display_df[columns_to_display])
        
        # Detailed Projects tab
        with analysis_tabs[3]:
            st.markdown("### Detailed Multi-College Projects")
            
            # Year selection
            years = sorted(yearly_stats['fiscal_year'].unique()) if not yearly_stats.empty else []
            if years:
                selected_year = st.selectbox(
                    "Select Fiscal Year",
                    options=years,
                    index=len(years)-1,  # Default to the most recent year
                    key="multi_college_year_select"
                )
                
                # Get multi-college projects for the selected year
                multi_college_projects = get_multi_college_projects_by_year(
                    st.session_state.filtered_dataframes, selected_year
                )
                
                # Display projects
                if not multi_college_projects.empty:
                    st.markdown(f"#### {len(multi_college_projects)} Multi-College Projects in FY {selected_year}")
                    
                    # Use the expandable project list component
                    from modules.components import create_expandable_project_list
                    
                    create_expandable_project_list(
                        projects=multi_college_projects,
                        title=f"Multi-College Projects - FY {selected_year}",
                        show_metrics=True,
                        unique_key="multi_college_projects_list"
                    )
                    
                    # Advanced data exploration
                    st.markdown("#### Advanced Data Exploration")
                    
                    from modules.components import project_details_view
                    
                    project_details_view(
                        projects=multi_college_projects,
                        title="Multi-College Projects Data Explorer",
                        unique_key="multi_college_projects_explorer"
                    )
                else:
                    st.info(f"No multi-college projects found for fiscal year {selected_year}")
            else:
                st.info("No fiscal year data available")
        
        # Funding Analysis tab
        with analysis_tabs[4]:
            st.markdown("### Funding Analysis")
            
            # Create funding comparison chart
            fig = create_multi_vs_single_college_comparison_chart(comparison_data)
            st.pyplot(fig)
            
            # Display funding data by year if available
            if not yearly_stats.empty and 'avg_award_multi_projects' in yearly_stats.columns:
                # Check if any funding data is available
                has_funding_data = (
                    yearly_stats['avg_award_multi_projects'].notna().any() or
                    yearly_stats['avg_award_single_projects'].notna().any()
                )
                
                if has_funding_data:
                    st.markdown("#### Funding Trends by Year")
                    
                    # Create funding trends chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Filter to years with funding data
                    funding_data = yearly_stats.dropna(subset=['avg_award_multi_projects', 'avg_award_single_projects'], how='all')
                    
                    if not funding_data.empty:
                        years = funding_data['fiscal_year'].astype(str)
                        
                        # Plot multi-college funding
                        if funding_data['avg_award_multi_projects'].notna().any():
                            ax.plot(years, funding_data['avg_award_multi_projects'], 
                                   marker='o', label='Multi-College Projects', 
                                   color='#1f77b4', linewidth=2)
                        
                        # Plot single-college funding
                        if funding_data['avg_award_single_projects'].notna().any():
                            ax.plot(years, funding_data['avg_award_single_projects'], 
                                   marker='s', label='Single-College Projects', 
                                   color='#ff7f0e', linewidth=2)
                        
                        # Set labels and title
                        ax.set_xlabel('Fiscal Year', fontsize=12)
                        ax.set_ylabel('Average Award Amount ($)', fontsize=12)
                        ax.set_title('Average Award Amount by Project Type', fontsize=14, fontweight='bold')
                        
                        # Format y-axis as currency
                        ax.yaxis.set_major_formatter('${x:,.0f}')
                        
                        # Add grid and legend
                        ax.grid(axis='y', linestyle='--', alpha=0.3)
                        ax.legend()
                        
                        # Rotate x-axis labels
                        plt.xticks(rotation=45)
                        
                        # Adjust layout
                        plt.tight_layout()
                        
                        # Display the chart
                        st.pyplot(fig)
                        
                        # Display data table
                        with st.expander("View Funding Data", expanded=False):
                            # Format the DataFrame for display
                            display_df = funding_data.copy()
                            
                            # Format currency values
                            for col in ['avg_award_all_projects', 'avg_award_multi_projects', 'avg_award_single_projects']:
                                if col in display_df.columns:
                                    display_df[col] = display_df[col].apply(
                                        lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A"
                                    )
                            
                            # Rename columns for better readability
                            display_df = display_df.rename(columns={
                                'fiscal_year': 'Fiscal Year',
                                'avg_award_all_projects': 'Avg. Award (All Projects)',
                                'avg_award_multi_projects': 'Avg. Award (Multi-College)',
                                'avg_award_single_projects': 'Avg. Award (Single-College)'
                            })
                            
                            # Select columns to display
                            columns_to_display = ['Fiscal Year', 'Avg. Award (All Projects)', 
                                                'Avg. Award (Multi-College)', 'Avg. Award (Single-College)']
                            
                            # Display the table
                            st.dataframe(display_df[columns_to_display])
                    else:
                        st.info("No funding data available by year")
                else:
                    st.info("No funding data available")
            else:
                st.info("No funding data available")
    
    # College Collaboration Network tab
    with tab7:
        st.markdown(f'<h2 class="sub-header">College Collaboration Network Analysis</h2>', unsafe_allow_html=True)
        
        # Create the college collaboration network
        adj_matrix, network_metrics = create_college_collaboration_network(
            st.session_state.filtered_dataframes
        )
        
        if adj_matrix.empty or not network_metrics:
            st.warning("No collaboration data available. Try uploading data or adjusting your filters.")
        else:
            # Get role distribution data
            role_data = get_college_role_distribution(st.session_state.filtered_dataframes)
            
            # Get diversity metrics
            diversity_data = get_college_collaboration_diversity(st.session_state.filtered_dataframes)
            
            # Get pairwise collaboration data
            pairs_data = get_pairwise_college_collaborations(st.session_state.filtered_dataframes)
            
            # Network Visualization Options
            viz_type = st.radio(
                "Visualization Type",
                ["Network Graph", "Collaboration Heatmap", "Role Distribution", "Collaboration Diversity", "Comprehensive Dashboard"],
                horizontal=True,
                index=0
            )
            
            # Display visualizations based on selection
            if viz_type == "Network Graph":
                top_n = st.slider("Number of top colleges to include", 5, 30, 15)
                st.markdown("### College Collaboration Network Graph")
                st.markdown("This graph shows the strength of collaboration between colleges. Larger nodes indicate colleges with more collaborations.")
                
                fig = create_network_graph(adj_matrix, network_metrics, top_n=top_n)
                st.pyplot(fig)
                
                # Display top collaborating pairs
                st.markdown("### Top Collaborating College Pairs")
                st.dataframe(pairs_data.head(10))
                
            elif viz_type == "Collaboration Heatmap":
                st.markdown("### College Collaboration Heatmap")
                st.markdown("This heatmap shows the strength of collaboration between pairs of colleges.")
                
                fig = create_heatmap(adj_matrix)
                st.pyplot(fig)
                
            elif viz_type == "Role Distribution":
                top_n = st.slider("Number of colleges to show", 5, 20, 10)
                st.markdown("### PI vs Co-PI Role Distribution by College")
                st.markdown("This chart shows the distribution of PI and Co-PI roles for each college.")
                
                fig = create_role_distribution_chart(role_data, top_n=top_n)
                st.pyplot(fig)
                
                # Display role distribution data
                st.dataframe(role_data)
                
            elif viz_type == "Collaboration Diversity":
                top_n = st.slider("Number of colleges to show", 5, 20, 10)
                st.markdown("### College Collaboration Diversity")
                st.markdown("This chart shows how diverse each college's collaboration network is.")
                
                fig = create_collaboration_diversity_chart(diversity_data, top_n=top_n)
                st.pyplot(fig)
                
                # Display diversity metrics
                st.dataframe(diversity_data)
                
            elif viz_type == "Comprehensive Dashboard":
                st.markdown("### Comprehensive College Collaboration Dashboard")
                
                fig = create_college_collaboration_dashboard(adj_matrix, network_metrics, role_data, diversity_data)
                st.pyplot(fig)
    
    # Export Data tab
    with tab8:
        st.markdown(f'<h2 class="sub-header">Export Data</h2>', unsafe_allow_html=True)
        
        # Create output directory
        output_dir = "output"
        ensure_directory_exists(output_dir)
        
        # Export options
        export_format = st.radio(
            "Export Format",
            ["Excel", "CSV"]
        )
        
        if export_format == "Excel":
            # Export all dataframes to Excel
            output_path = os.path.join(output_dir, "grant_data_analysis.xlsx")
            
            if st.button("Export to Excel"):
                with st.spinner("Exporting data..."):
                    export_to_excel(st.session_state.filtered_dataframes, output_path)
                    st.success(f"Data exported to {output_path}")
                    
                    # Provide download link
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Excel File",
                            data=file,
                            file_name="grant_data_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        else:
            # Export individual dataframes to CSV
            st.markdown("Select data to export:")
            
            for sheet_name, df in st.session_state.filtered_dataframes.items():
                output_path = os.path.join(output_dir, f"{sheet_name.lower()}.csv")
                
                if st.button(f"Export {sheet_name} to CSV"):
                    with st.spinner(f"Exporting {sheet_name}..."):
                        export_to_csv(df, output_path)
                        st.success(f"{sheet_name} exported to {output_path}")
                        
                        # Provide download link
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label=f"Download {sheet_name} CSV",
                                data=file,
                                file_name=f"{sheet_name.lower()}.csv",
                                mime="text/csv"
                            )
else:
    # Display instructions when no file is uploaded
    st.markdown("""
    <div class="info-box">
        <h2>Welcome to the SDSU Grant Data Explorer</h2>
        <p>This application allows you to analyze and visualize SDSU grant award data.</p>
        <h3>How to use:</h3>
        <ol>
            <li>Upload an Excel file containing grant data using the file uploader in the sidebar.</li>
            <li>Select the sheets to analyze (default: AwardsRawData and AwardsCoPIsRawData).</li>
            <li>Use the filter options to narrow down the data by fiscal year.</li>
            <li>Explore the different tabs to view various analyses and visualizations.</li>
            <li>Export the data or visualizations as needed.</li>
        </ol>
        <h3>Required Data Format:</h3>
        <p>The Excel file should contain the following columns:</p>
        <ul>
            <li>Grant Code</li>
            <li>Fiscal Year</li>
            <li>PI</li>
            <li>Co-PI(s)</li>
            <li>CollegeUnit</li>
            <li>Department</li>
        </ul>
    </div>
    """, unsafe_allow_html=True) 