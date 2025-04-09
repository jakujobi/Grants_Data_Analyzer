# Projects with X Co-PIs Feature Documentation

The "Projects with X Co-PIs" feature provides powerful analysis capabilities for exploring grant projects based on the number of Co-Principal Investigators (Co-PIs) involved. This document explains how to use this feature and how to customize it for specific analysis needs.

## Overview

This feature allows users to:

1. Filter projects based on Co-PI count conditions (exactly, more than, less than, etc.)
2. View yearly trends and statistics
3. Perform detailed year-by-year analysis
4. Use interactive data views with search and export capabilities

## Feature Location

The "Projects with X Co-PIs" feature is available as a dedicated tab in the main application interface. After loading data, navigate to this tab to access the feature.

## Analysis Types

The feature offers three main analysis types:

### 1. Yearly Trend

This view provides a year-by-year summary of projects matching the specified Co-PI condition. It displays:

- Bar charts showing the selected metric over time
- Comparison charts showing matching projects vs. total projects
- Summary statistics for the entire time period
- Tabular data for detailed review

### 2. Year-by-Year Details

This view focuses on a specific fiscal year and provides:

- A detailed table of projects meeting the criteria for the selected year
- Expandable sections for each project with comprehensive details
- Information about PIs, Co-PIs, and College Units for each project

### 3. Detailed Project View

This view provides advanced data exploration capabilities:

- Flexible year filtering (All Years, Year Range, Single Year)
- Summary metrics (Total Projects, Total PIs, Total Co-PIs)
- Expandable project list with two-column layout for project details
- Interactive data exploration with search, pagination, multiple display modes, and export options

## Filtering Options

### Co-PI Count Condition

The feature supports the following conditions for filtering projects:

| Condition | Description | Example |
|-----------|-------------|---------|
| At Least | Projects with at least X Co-PIs | At least 2 Co-PIs |
| At Most | Projects with at most X Co-PIs | At most 3 Co-PIs |
| Exactly | Projects with exactly X Co-PIs | Exactly 1 Co-PI |
| More Than | Projects with more than X Co-PIs | More than 0 Co-PIs |
| Less Than | Projects with fewer than X Co-PIs | Less than 2 Co-PIs |

### Visualization Metric

When viewing the "Yearly Trend" analysis, you can choose from the following metrics to visualize:

- **Matching Projects Count**: Number of projects matching the Co-PI criteria per year
- **Matching Projects Percentage**: Percentage of projects matching the criteria out of all projects
- **Total Co-PIs Count**: Total number of Co-PIs across all matching projects

## Using the Feature

### Step-by-Step Guide

1. Upload your data using the file uploader in the sidebar
2. Navigate to the "Projects with X Co-PIs" tab
3. Select a Co-PI count condition (e.g., "Exactly")
4. Enter the number of Co-PIs to filter by (e.g., "0" for projects with no Co-PIs)
5. Choose a visualization metric (for trend analysis)
6. Select an analysis type (Yearly Trend, Year-by-Year Details, or Detailed Project View)
7. If needed, specify additional filters (e.g., fiscal year range)
8. Explore the results through the various visualizations and data tables

### Example Analyses

#### Example 1: Finding Solo PI Projects (No Co-PIs)

1. Set Co-PI Count Condition to "Exactly"
2. Enter "0" as the Number of Co-PIs
3. Select "Yearly Trend" to see how solo PI projects have changed over time
4. Use the "Matching Projects Percentage" metric to see the proportion of solo projects

#### Example 2: Highly Collaborative Projects

1. Set Co-PI Count Condition to "More Than"
2. Enter "3" as the Number of Co-PIs (to find projects with 4+ Co-PIs)
3. Select "Detailed Project View" to explore these highly collaborative projects
4. Use the search functionality to look for specific colleges or investigators

#### Example 3: Year-over-Year Comparison

1. Set Co-PI Count Condition to "At Least"
2. Enter "1" as the Number of Co-PIs (to find all projects with Co-PIs)
3. Select "Yearly Trend" and view the "Comparison Chart"
4. Observe how the proportion of projects with Co-PIs has changed over time

## Technical Details

### Underlying Functions

The feature uses the following functions from the `analysis.py` module:

- `get_projects_by_co_pi_count`: Gets projects matching the specified Co-PI condition
- `get_projects_by_co_pi_count_yearly`: Gets yearly statistics for projects matching the Co-PI condition
- `get_project_details_by_co_pi_filter`: Gets detailed project information for a specific year

And the following visualization functions from the `visuals.py` module:

- `create_co_pi_analysis_chart`: Creates a bar chart for the selected metric
- `create_co_pi_comparison_chart`: Creates a comparison chart (matching vs. total projects)

For interactive data views, it uses components from the `components.py` module:

- `create_expandable_project_list`: Displays projects in an expandable list format
- `project_details_view`: Provides an interactive data exploration interface

### Data Structure

The feature operates on DataFrames with the following structure:

- `grant_code`: Unique identifier for each grant
- `fiscal_year`: The fiscal year of the grant
- `pi`: Principal Investigator(s)
- `co_pi_list`: List of Co-Principal Investigators
- `co_pi_count`: Number of Co-PIs (calculated)
- `college_units`: List of college units involved
- `award_amount`: Grant award amount (if available)

## Customizing the Feature

If you wish to customize this feature for your specific needs, here are some suggestions:

### Adding New Metrics

To add a new metric for visualization:

1. Add the metric calculation to the `get_projects_by_co_pi_count_yearly` function in `analysis.py`
2. Update the metric options dictionary in the UI code in `streamlit_app.py`
3. Add handling for the new metric in the `create_co_pi_analysis_chart` function in `visuals.py`

### Adding New Filtering Options

To add a new filtering condition:

1. Update the condition options dictionary in the UI code in `streamlit_app.py`
2. Add handling for the new condition in the `get_projects_by_co_pi_count` function in `analysis.py`

### Extending the UI

To add new UI components or integrations:

1. Create new UI components in the `components.py` module if needed
2. Add the components to the relevant section in the `streamlit_app.py` file
3. Update this documentation to reflect the changes

## Troubleshooting

### No Data Available

If you see "No data available" messages:

1. Make sure you have uploaded a valid Excel file with the required columns
2. Check that the selected sheets contain the necessary data (PI, Co-PI information)
3. Verify that the fiscal year filter (if applied) includes years present in your data
4. Ensure the Co-PI condition is appropriate (e.g., if filtering for "Exactly 5 Co-PIs" but no projects have exactly 5 Co-PIs)

### Performance Issues

If the feature is slow:

1. Consider reducing the amount of data loaded (select only necessary sheets)
2. Filter by a specific year range rather than analyzing all years
3. Use the "Year-by-Year Details" or "Detailed Project View" for more focused analysis

## Future Enhancements

Planned future enhancements for this feature include:

1. **Network Visualization**: Visual representation of PI/Co-PI collaboration networks
2. **Statistical Analysis**: Advanced statistical measures for collaboration patterns
3. **Time Series Analysis**: Trend analysis and forecasting for collaboration metrics
4. **Export Options**: Additional export formats and customization options
5. **Comparative Analysis**: Side-by-side comparison of different Co-PI conditions 