# SDSU Grant Data Explorer

A Streamlit application for analyzing and visualizing SDSU grant award data.

## Features

- Upload and analyze Excel files containing grant data
- Filter data by fiscal year
- View summary statistics and metrics
- Analyze multi-college projects and collaborations
- Track project counts by person and college
- Export data in Excel or CSV format
- Interactive visualizations with SDSU branding

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
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

- `app.py`: Main Streamlit application
- `modules/`: Python modules containing core functionality
  - `load_data.py`: Functions for loading and validating Excel files
  - `clean_data.py`: Functions for cleaning and normalizing data
  - `analysis.py`: Functions for analyzing grant data
  - `visuals.py`: Functions for creating visualizations
  - `utils.py`: Utility functions
- `output/`: Directory for exported files
- `requirements.txt`: Required Python packages

## Development

To contribute to this project:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 