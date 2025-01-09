import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
method_tag = "scaled_zscore" 
file_tag = "class_ny_arrests"
filename_train = f"Data/{file_tag}_smote.csv"  # Replace with your dataset's filename
filename_test = f"Data/{file_tag}_{method_tag}.csv"
original_data_train = pd.read_csv(filename_train)
original_data_test = pd.read_csv(filename_test)

# Get only 10% of the data
train_data = original_data_train.sample(frac=0.1, random_state=42)
test_data = original_data_test.sample(frac=0.1, random_state=42)

# Save the splits to separate CSV files
train_data.to_csv(f"Data/train_{file_tag}_sample_smote.csv", index=False)
test_data.to_csv(f"Data/test_{file_tag}_sample_zscore.csv", index=False)
