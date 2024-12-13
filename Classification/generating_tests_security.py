import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
method_tag = "scaled_zscore" 
file_tag = "class_ny_arrests"
#filename = f"Data/{file_tag}_{method_tag}.csv"  # Replace with your dataset's filename
#data = pd.read_csv(filename)

# Drop the JURISDICTION_CODE column
#data = data.drop(columns=["JURISDICTION_CODE"])

# Get only 10% of the data
#data = data.sample(frac=0.05, random_state=42)

# Split the data into 70% and 30%
#train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

'''
test_file = pd.read_csv(f"Data/test_{file_tag}_{method_tag}.csv")
test_file = test_file.sample(frac=0.05, random_state=42)
test_file.to_csv(f"Data/test_{file_tag}_{method_tag}.csv", index=False)
'''
# Save the splits to separate CSV files
#train_data.to_csv(f"Data/train_{file_tag}_{method_tag}.csv", index=False)
#test_data.to_csv(f"Data/test_{file_tag}_{method_tag}.csv", index=False)
