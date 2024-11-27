from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types

filename = "Data/class_ny_arrests.csv"
file_tag = "class_ny_arrests"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
print(f"Original data: {data.shape}")

df: DataFrame = data.dropna(how="any", inplace=False)

print(f"Final data: {df.shape}")

result = df[(df['PD_CD'].notna()) & (df['KY_CD'].isna())]
print(result.shape)

df.to_csv(f"./Data/{file_tag}_missing_values.csv", index=True)