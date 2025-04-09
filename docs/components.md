# UI Components Documentation

The `components.py` module provides reusable UI components for the Grant Data Explorer application. These components can be easily integrated into any part of the application to provide consistent, feature-rich interfaces for data visualization and exploration.

## Available Components

### 1. `detailed_data_view`

A versatile, highly configurable data view component that provides rich data exploration capabilities.

#### Features:
- Multiple display modes (Tabular, Cards, Expandable)
- Pagination with customizable page size
- Search functionality across text columns
- Export options for current page or all results
- Customizable column formatting

#### Usage:

```python
from modules.components import detailed_data_view

# Basic usage
detailed_data_view(
    data=my_dataframe,
    title="My Data View",
    unique_key="my_data_view"
)

# Advanced usage with all options
detailed_data_view(
    data=my_dataframe,
    title="Detailed Analysis",
    key_columns=['id', 'name'],  # Columns to use as identifiers in expander headers
    formatter={
        'amount': lambda x: f"${x:,.2f}",
        'date': lambda x: x.strftime("%Y-%m-%d")
    },
    page_size=15,  # Show 15 records per page
    allow_export=True,
    unique_key="detailed_analysis"
)
```

#### Parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `data` | pd.DataFrame | DataFrame containing the data to display | (Required) |
| `title` | str | Title to display above the data view | "Detailed Data View" |
| `key_columns` | List[str] | Columns to use as identifiers in headers | None |
| `formatter` | Dict[str, Callable] | Functions to format specific columns | None |
| `page_size` | int | Number of records per page | 10 |
| `allow_export` | bool | Whether to show export options | True |
| `unique_key` | str | Unique key for this component instance | "detailed_view" |

### 2. `project_details_view`

A specialized view for project data that builds on `detailed_data_view` with project-specific configurations.

#### Features:
- All features of `detailed_data_view`
- Pre-configured for project data with appropriate formatting
- Proper handling of grant codes and fiscal years as identifiers

#### Usage:

```python
from modules.components import project_details_view

project_details_view(
    projects=projects_dataframe,
    title="Grant Projects",
    unique_key="grant_projects"
)
```

#### Parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `projects` | pd.DataFrame | DataFrame containing project data | (Required) |
| `title` | str | Title to display above the data view | "Project Details" |
| `unique_key` | str | Unique key for this component instance | "project_details" |

### 3. `create_expandable_project_list`

A component for displaying a list of projects with expandable details for each project.

#### Features:
- Summary metrics (total projects, PIs, Co-PIs)
- Expandable sections for each project
- Two-column layout for project details
- Special handling for list fields like PIs, Co-PIs, and college units

#### Usage:

```python
from modules.components import create_expandable_project_list

create_expandable_project_list(
    projects=projects_dataframe,
    fiscal_year=2022,  # Optional: Filter by specific year
    title="2022 Projects",
    show_metrics=True,
    unique_key="projects_2022"
)
```

#### Parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `projects` | pd.DataFrame | DataFrame containing project data | (Required) |
| `fiscal_year` | Optional[int] | Fiscal year to filter by | None |
| `title` | str | Title to display above the project list | "Projects" |
| `show_metrics` | bool | Whether to show summary metrics | True |
| `unique_key` | str | Unique key for this component instance | "project_list" |

## Integration Examples

### Example 1: Adding to a New Tab

```python
import streamlit as st
from modules.components import create_expandable_project_list, project_details_view

def render_projects_tab():
    st.title("Project Analysis")
    
    # Get project data
    projects = get_projects_data()  # Your function to fetch project data
    
    # Display projects as an expandable list
    create_expandable_project_list(
        projects=projects,
        title="Current Projects",
        unique_key="current_projects"
    )
    
    # Add a divider
    st.markdown("---")
    
    # Add advanced data exploration
    st.subheader("Advanced Data Exploration")
    project_details_view(
        projects=projects,
        title="Project Database",
        unique_key="project_database"
    )
```

### Example 2: Using with Filters

```python
import streamlit as st
from modules.components import detailed_data_view

def render_filtered_data_view():
    # Create filter controls
    st.subheader("Filter Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filter_type = st.selectbox(
            "Filter By",
            options=["Department", "Year", "Amount"]
        )
    
    with col2:
        if filter_type == "Department":
            filter_value = st.selectbox("Department", ["Engineering", "Science", "Arts"])
        elif filter_type == "Year":
            filter_value = st.slider("Year", 2010, 2023)
        else:
            filter_value = st.number_input("Minimum Amount", 0, 1000000, 10000)
    
    # Get and filter data
    data = get_data()  # Your function to fetch data
    filtered_data = apply_filter(data, filter_type, filter_value)  # Your filtering function
    
    # Display the filtered data
    detailed_data_view(
        data=filtered_data,
        title=f"Data filtered by {filter_type}",
        unique_key=f"filtered_data_{filter_type.lower()}"
    )
```

## Best Practices

1. **Unique Keys**: Always provide a unique `unique_key` parameter when using multiple instances of the same component on a page.

2. **Formatters**: For complex data types or values that need special formatting, provide custom formatter functions.

3. **Key Columns**: Choose meaningful key columns for the expander headers to make the data more navigable.

4. **Pagination**: Adjust the `page_size` parameter based on the complexity of your data. For data with many columns or complex values, use a smaller page size.

5. **Data Preparation**: Ensure your data is properly structured before passing it to the components. In particular:
   - Make sure list fields are actual Python lists, not string representations
   - Clean up any NaN values or convert them to appropriate placeholders
   - Convert date columns to proper datetime objects

## Extending Components

These components are designed to be extensible. If you need custom functionality:

1. Use the existing components as a base and extend them
2. Create a new component that combines multiple existing components
3. Submit a pull request to add your component to the module

Example of extending a component:

```python
def my_custom_data_view(data, title="Custom View", **kwargs):
    """
    A custom data view with additional preprocessing.
    """
    # Preprocess data
    processed_data = preprocess_my_data(data)
    
    # Add custom formatters
    formatters = kwargs.get('formatter', {})
    formatters.update({
        'custom_field': my_custom_formatter
    })
    
    # Call the base component with processed data and updated kwargs
    kwargs['formatter'] = formatters
    detailed_data_view(
        data=processed_data,
        title=title,
        **kwargs
    )
``` 