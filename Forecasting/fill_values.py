import pandas as pd

# Load the dataset
file_path = "Data/Time_Series/forecast_gdp_europe_first_diff_test.csv"  # Update with your file's path
data = pd.read_csv(file_path)

# Display initial information
print("Initial Missing Values:\n", data.isnull().sum())

# Step 1: Apply forward fill for initial continuity
data_ffill = data.fillna(method='ffill')

# Step 2: Apply linear interpolation for smooth imputation in numeric data
data_imputed = data_ffill.interpolate(method='linear')

# Step 3: Apply forward fill and backfill to ensure no trailing NaN values
data_final_imputed = data_imputed.fillna(method='ffill').fillna(method='bfill')

# Verify that no missing values remain
print("Final Missing Values:\n", data_final_imputed.isnull().sum())

# Save the imputed dataset
output_path = "Data/Time_Series/forecast_gdp_europe_no_nan_first_diff_test.csv"
data_final_imputed.to_csv(output_path, index=False)

print(f"Imputed dataset saved to {output_path}")

