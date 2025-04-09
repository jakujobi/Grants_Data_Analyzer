# SDSU Grant Data Explorer

A Streamlit application for analyzing and visualizing SDSU grant award data.

## Features

- Upload and analyze Excel files containing grant data
- Filter data by fiscal year
- View summary statistics and metrics
- Analyze multi-college projects and collaborations
- Track project counts by person and college
- Analyze projects based on Co-PI counts with detailed data exploration
- Interactive data views with search, filtering, and export capabilities
- Export data in Excel or CSV format
- Interactive visualizations with SDSU branding
- User-configurable file size limits and security controls

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Upload an Excel file containing grant data using the file uploader in the sidebar
3. Select the sheets to analyze (default: AwardsRawData and AwardsCoPIsRawData)
4. Use the filter options to narrow down the data by fiscal year
5. Explore the different tabs to view various analyses and visualizations
6. Export the data or visualizations as needed

## Required Data Format

The Excel file should contain the following columns:
- Grant Code
- Fiscal Year
- PI
- Co-PI(s)
- CollegeUnit
- Department

## Project Structure

- `streamlit_app.py`: Main Streamlit application
- `modules/`: Python modules containing core functionality
  - `load_data.py`: Functions for loading and validating Excel files
  - `clean_data.py`: Functions for cleaning and normalizing data
  - `analysis.py`: Functions for analyzing grant data
  - `visuals.py`: Functions for creating visualizations
  - `utils.py`: Utility functions
  - `components.py`: Reusable UI components for interactive data visualization
- `output/`: Directory for exported files
- `requirements.txt`: Required Python packages

## Key Features in Detail

### Summary Statistics
View comprehensive summary statistics about grants, including counts of unique PIs, Co-PIs, and colleges.

### Multi-College Projects
Analyze projects that involve collaboration between multiple colleges and identify projects with multiple Co-PIs.

### Collaborators
Identify colleges with the most collaborators and view collaboration networks.

### Project Counts
View project counts by person (PI or Co-PI) and filter by role.

### College Analysis
Analyze project distribution across colleges using various visualizations.

### Projects with X Co-PIs
A powerful analysis tool that allows you to:
- Filter projects based on Co-PI count conditions (exactly, at least, at most, etc.)
- View yearly trends and statistics
- Explore detailed project information
- Use interactive data views with search and export capabilities

### Data Export
Export analyzed data in Excel or CSV format for further analysis.

## Security and Configuration

The application includes several security features:
- Configurable maximum file upload size (default: 50MB)
- Validation of uploaded file types
- Protection against excessive resource usage
- Temporary file cleanup

## Development

To contribute to this project:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 