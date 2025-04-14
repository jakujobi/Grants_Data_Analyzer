# Multi-College Projects Analyzer Documentation

The Multi-College Projects Analyzer is a powerful tool for analyzing and visualizing projects where Principal Investigators (PIs) and Co-Principal Investigators (Co-PIs) come from multiple colleges. This feature helps identify trends in cross-college collaboration, analyze funding differences, and explore detailed project information.

## What is a Multi-College Project?

In this analyzer, a project is considered a **multi-college project** when the PI and/or Co-PIs collectively represent more than one college. For example:

- **Project A:** PI from Engineering, Co-PI from Natural Sciences → *Multi-college (2 colleges)*
- **Project B:** PI and all Co-PIs from Engineering → *Not multi-college (1 college)*
- **Project C:** PI from Engineering, one Co-PI from Natural Sciences, another Co-PI from Engineering → *Multi-college (2 colleges)*
- **Project D:** PI from Engineering, one Co-PI from Natural Sciences, one Co-PI from Engineering, one Co-PI from Nursing → *Multi-college (3 colleges)*

## Feature Overview

The Multi-College Projects Analyzer provides five main analysis views:

1. **Dashboard:** A comprehensive overview of multi-college project statistics and visualizations
2. **Yearly Trends:** Analysis of how multi-college collaborations have changed over time
3. **College Comparisons:** Rankings and metrics about which colleges engage in the most cross-college collaboration
4. **Detailed Projects:** Individual project information for multi-college projects
5. **Funding Analysis:** Comparison of funding amounts between multi-college and single-college projects

## Using the Multi-College Projects Analyzer

### Accessing the Analyzer

1. Upload your data using the file uploader in the sidebar
2. Navigate to the "Multi-College Projects Analyzer" tab in the main application
3. The data will be automatically analyzed to identify multi-college projects
4. Use the tabs at the top of the analyzer section to switch between different analysis views

### Dashboard View

The Dashboard provides a high-level overview of multi-college project statistics:

- **Summary Metrics:** Total multi-college projects, percentage of all projects, average colleges per project, and funding differences
- **Trend Chart:** Shows the number and percentage of multi-college projects over time
- **Colleges per Project:** Shows the average number of colleges involved in multi-college projects over time
- **Project Distribution:** Pie chart showing the distribution of multi-college vs. single-college projects
- **Funding Comparison:** Bar chart comparing average award amounts between multi-college and single-college projects (if funding data is available)

You can export the dashboard as an image for presentations or reports by clicking the "Export Dashboard" button.

### Yearly Trends View

The Yearly Trends view focuses on temporal patterns in multi-college collaboration:

- **Trend Chart:** Shows both the count and percentage of multi-college projects for each fiscal year
- **Data Table:** Expandable section showing detailed yearly statistics

This view helps identify trends, such as whether cross-college collaboration is increasing or decreasing over time.

### College Comparisons View

The College Comparisons view reveals which colleges are most active in multi-college collaboration:

- **College Ranking Chart:** Horizontal bar chart showing colleges ranked by their participation in multi-college projects
- **College Metrics:** Expandable table with detailed metrics for each college

You can adjust the number of colleges shown in the chart using the slider control.

### Detailed Projects View

The Detailed Projects view allows you to explore individual multi-college projects:

1. Select a fiscal year from the dropdown menu
2. View the list of multi-college projects for that year
3. Expand project details to see information about PIs, Co-PIs, and colleges
4. Use the Advanced Data Explorer for more sophisticated filtering and searching

This view integrates the reusable components from the `components.py` module for consistent user experience.

### Funding Analysis View

The Funding Analysis view compares funding between multi-college and single-college projects:

- **Comparison Chart:** Side-by-side comparison of project counts and average award amounts
- **Funding Trends:** Line chart showing how funding has changed over time for both project types
- **Funding Data:** Expandable table with detailed yearly funding statistics

## Key Metrics Explained

The Multi-College Projects Analyzer calculates several key metrics:

- **Multi-College Percentage:** The percentage of all projects that involve multiple colleges
- **Average Colleges per Project:** The average number of different colleges represented in multi-college projects
- **Funding Difference:** The percentage difference in average award amounts between multi-college and single-college projects
- **College Collaboration Metrics:** For each college, statistics on multi-college participation and collaboration partners

## Technical Implementation

The analyzer is built using several modular components:

### Analysis Functions

- `identify_multi_college_projects`: Identifies projects where PIs and Co-PIs come from multiple colleges
- `get_multi_college_project_yearly_stats`: Calculates yearly statistics for multi-college projects
- `get_college_collaboration_metrics`: Computes collaboration metrics for each college
- `get_multi_college_vs_single_college_comparison`: Compares metrics between multi-college and single-college projects
- `get_multi_college_projects_by_year`: Retrieves multi-college projects for a specific fiscal year

### Visualization Functions

- `create_multi_college_trend_chart`: Creates a chart showing the trend of multi-college projects over time
- `create_multi_vs_single_college_comparison_chart`: Creates a comparison chart for multi-college and single-college projects
- `create_college_collaboration_chart`: Creates a chart showing colleges with the most multi-college collaborations
- `create_avg_colleges_per_project_chart`: Creates a chart showing the average number of colleges per multi-college project
- `create_multi_college_dashboard`: Creates a comprehensive dashboard for multi-college project analysis

### User Interface Components

The analyzer uses reusable UI components from the `components.py` module:

- `create_expandable_project_list`: Creates an expandable list of projects with detailed information
- `project_details_view`: Provides an interactive data exploration interface with search and pagination

## Customizing the Analysis

If you wish to customize the Multi-College Projects Analyzer, you can adjust the following:

- **Top N colleges:** Control how many colleges are shown in the college comparisons chart
- **Year selection:** Filter the analysis to focus on specific fiscal years
- **Export options:** Save visualizations and data for further analysis or reporting

## Troubleshooting

If you encounter issues with the Multi-College Projects Analyzer:

1. **No data appears:** Ensure your Excel file has columns for Grant Code, Fiscal Year, PI, Co-PI(s), and CollegeUnit
2. **College information is missing:** Verify that the CollegeUnit column contains valid college names
3. **Funding analysis is missing:** Ensure your data includes an award_amount column
4. **Year trends show unexpected patterns:** Check for data quality issues like missing years or inconsistent formatting

## Future Enhancements

Planned enhancements for this feature include:

1. **Network visualizations:** Interactive network graphs showing collaboration between colleges
2. **Predictive analysis:** Identifying potential future collaboration opportunities
3. **Success metrics:** Analyzing whether multi-college projects have higher success rates
4. **Publication analysis:** Integration with publication data to measure research output from multi-college projects
``` 