import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
method_tag = "scaled_minmax" 
file_tag = "class_ny_arrests"
filename = f"Data/{file_tag}_{method_tag}.csv"  # Replace with your dataset's filename
data = pd.read_csv(filename, na_values="", index_col="ARREST_KEY")

for i in range(0, 10):
   print("Generating split number:", i)
   
   to_keep = data.sample(frac = 0.01)
   # Split the data into 70% training and 30% testing
   train_data, test_data = train_test_split(to_keep, test_size=0.3, random_state=42)
   # Save the splits to separate CSV files
   train_data.to_csv(f"Data/Scaling/MinMax/train_{file_tag}_{method_tag}_run{i}.csv", index=False)
   test_data.to_csv(f"Data/Scaling/MinMax/test_{file_tag}_{method_tag}_run{i}.csv", index=False)

#print("Data has been split and saved:")
#print("70% of data saved to 'train_data.csv'")
#print("30% of data saved to 'test_data.csv'")



    # Remove column Unnamed: 0 from the dataset
    #filename = f"Data/Missing_Values/Method2/test_{file_tag}_{method_tag}_run{i}.csv"
    #data = pd.read_csv(filename, na_values="", index_col=None)
#
    #data = data.drop(columns=["Unnamed: 0"])
    #
    #data.to_csv(filename, index=False)