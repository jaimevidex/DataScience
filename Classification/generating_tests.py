import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
method_tag = "smote" 
file_tag = "class_financial_distress"
filename = f"Data/{file_tag}_{method_tag}.csv"  # Replace with your dataset's filename
data = pd.read_csv(filename)

# Split the data into 80% and 20%
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the splits to separate CSV files
train_data.to_csv(f"Data/train_{file_tag}_{method_tag}.csv", index=False)
test_data.to_csv(f"Data/test_{file_tag}_{method_tag}.csv", index=False)

print("Data has been split and saved:")
print("80% of data saved to 'train_data.csv'")
print("20% of data saved to 'test_data.csv'")
