from pandas import read_csv, DataFrame
import pandas as pd
from dslabs_functions import get_variable_types

file_tag = "class_ny_arrests"
filename = f"Data/{file_tag}_encoding.csv"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
print(f"Original data: {data.shape}")

# Filter how to get the column PD_CD with missing values to know the count 

removable_missing_values: DataFrame = data[(data["PD_CD"].isnull() & data["KY_CD"].isnull())]
print(f"Removable missing values: {removable_missing_values.shape}")

missing_values: DataFrame = data[(data["PD_CD"].notnull() & data["KY_CD"].isnull())]
print(f"Recoverable Missing values: {missing_values.shape}")

# Remove missing values impossible to obtain
data.drop(removable_missing_values.index, axis=0, inplace=True)

# dataframe with no missing values
data_no_missing_values: DataFrame = data.dropna()
print(f"Data after dropping missing values: {data_no_missing_values.shape}")

# Function to get the first KY_CD, PD_DESC, and OFNS_DESC values for the specificied PD_CD value
#def get_first_values(data: DataFrame, value: int) -> DataFrame:
#    firstValue = data[data_no_missing_values["PD_CD"] == value].head(1)
#    return firstValue[["KY_CD", "PD_DESC", "OFNS_DESC"]]
#
## For all the missing values get the first KY_CD, PD_DESC, and OFNS_DESC values and substitute them
#for index, row in missing_values.iterrows():
#    first_values: DataFrame = get_first_values(data_no_missing_values, row["PD_CD"])
#    if first_values.empty:
#        continue
#    missing_values.at[index, "KY_CD"] = first_values["KY_CD"].values[0]
#    missing_values.at[index, "PD_DESC"] = first_values["PD_DESC"].values[0]
#    missing_values.at[index, "OFNS_DESC"] = first_values["OFNS_DESC"].values[0]
#
#missing_values.dropna(inplace=True)
#print(f"Missing values recovered: {missing_values.shape}")
#
## Concatenate the data with missing values recovered with the data without missing values
#data_no_missing_values = pd.concat([data_no_missing_values, missing_values])
#print(f"Data after recovering missing values: {data_no_missing_values.shape}")

# Save the data with missing values recovered
data_no_missing_values.to_csv(f"./Data/{file_tag}_missing_values.csv", index=True)

