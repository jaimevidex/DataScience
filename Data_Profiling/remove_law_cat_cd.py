from pandas import read_csv, DataFrame
import os

# Load the data and print the number of variables and records
filename = "Data/Previous/class_ny_arrests.csv"
data: DataFrame = read_csv(filename, index_col=None)

# Remove the law_cat_cd column
data['CLASS'] = data['JURISDICTION_CODE'].apply(lambda x: 'NY' if x < 3 else 'nonNY')
data = data.drop("JURISDICTION_CODE", axis=1)

data.to_csv("Data/Previous/class_ny_arrests.csv", index=False)
