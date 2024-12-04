import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
method_tag = "scaled_zscore" 
file_tag = "class_ny_arrests"
filename = f"Data/{file_tag}_{method_tag}.csv"  # Replace with your dataset's filename
data = pd.read_csv(filename)

to_delete, to_keep = train_test_split(data, test_size=0.05, random_state=42)

# Split the data into 70% training and 30% testing
train_data, test_data = train_test_split(to_keep, test_size=0.3, random_state=42)

# Save the splits to separate CSV files
train_data.to_csv(f"Data/train_{file_tag}_{method_tag}.csv", index=False)
test_data.to_csv(f"Data/test_{file_tag}_{method_tag}.csv", index=False)

#print("Data has been split and saved:")
#print("70% of data saved to 'train_data.csv'")
#print("30% of data saved to 'test_data.csv'")