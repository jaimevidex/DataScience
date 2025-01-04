import pandas as pd

# Load the CSV file
input_file = "Data/Time_Series/forecast_gdp_europe.csv"  # Replace with your input file path
output_file = "Data/Time_Series/forecast_gdp_europe_no_nan.csv"  # Replace with your output file path

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Drop rows with any empty cells
df_cleaned = df.dropna()

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv(output_file, index=False)

print(f"Cleaned CSV saved to {output_file}")
