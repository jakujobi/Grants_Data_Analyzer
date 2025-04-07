import pandas as pd
import os

# Path to the sample file
sample_file = "sample_file.xlsx"

# Check if file exists
if not os.path.exists(sample_file):
    print(f"Error: {sample_file} not found.")
    exit(1)

# Get sheet names
print(f"Exploring {sample_file}...")
xl = pd.ExcelFile(sample_file)
sheet_names = xl.sheet_names
print(f"Sheet names: {sheet_names}")

# Check if required sheets exist
required_sheets = ["AwardsRawData", "AwardsCoPIsRawData"]
missing_sheets = [sheet for sheet in required_sheets if sheet not in sheet_names]

if missing_sheets:
    print(f"Warning: Missing required sheets: {missing_sheets}")
    print("Available sheets will be used instead.")

# Load and examine each sheet
for sheet in sheet_names:
    print(f"\n--- Sheet: {sheet} ---")
    df = pd.read_excel(sample_file, sheet_name=sheet)
    
    # Display basic information
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Display first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Check for key columns
    key_columns = ["Grant Code", "Fiscal Year", "PI", "Co-PI(s)", "CollegeUnit", "Department"]
    missing_columns = [col for col in key_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\nMissing key columns: {missing_columns}")
    else:
        print("\nAll key columns present.")
    
    # Check for data types
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\nMissing values per column:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.") 