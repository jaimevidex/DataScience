import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
filename = "Data/class_financial_distress_under.csv"  # Replace with your dataset's filename
data = pd.read_csv(filename)

# Split the data into 70% training and 30% testing
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Save the splits to separate CSV files
train_data.to_csv("Data/train_data_under_economical.csv", index=False)
test_data.to_csv("Data/test_data_under_economical.csv", index=False)

print("Data has been split and saved:")
print("70% of data saved to 'train_data.csv'")
print("30% of data saved to 'test_data.csv'")
