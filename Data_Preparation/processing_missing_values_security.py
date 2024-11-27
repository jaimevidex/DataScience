from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types

filename = "Data/class_ny_arrests.csv"
file_tag = "class_ny_arrests"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
print(f"Original data: {data.shape}")

# Filter how to get the column PD_CD with missing values to know the count 

missing_values: DataFrame = data[(data["PD_CD"].notnull() & data["KY_CD"].isnull())]
print(f"Missing values: {missing_values.shape}")

df: DataFrame = data.dropna(how="any", inplace=False)

print(f"Final data: {df.shape}")

# Fill the missing values with the values from other registries with the same PD_CD 
# Function to get the first row without none is this PD_CD
def get_first_row_without_none(data: DataFrame, pd_cd: int) -> DataFrame:
    print(data[data["PD_CD"] == pd_cd]["KY_CD"])
    return data[data["PD_CD"] == pd_cd].iloc[0]

#data["KY_CD"].fillna(get_first_row_without_none(df, missing_values["PD_CD"].iloc[0])["KY_CD"], inplace=True)

for ind in missing_values.index:
    print(get_first_row_without_none(df, missing_values.at[ind, "PD_CD"])["KY_CD"])
    #missing_values.at[ind, "KY_CD"] = get_first_row_without_none(df, missing_values.at[ind, "PD_CD"])["KY_CD"]

missing_values.to_csv(f"./Data/{file_tag}_missing_values.csv", index=True)

#missing_values: DataFrame = data[data["KY_CD"].isnull()]
#print(f"Missing values final: {missing_values.shape}")
#print(f"Final data: {data.shape}")
#
#data.to_csv(f"./Data/{file_tag}_missing_values.csv", index=True)