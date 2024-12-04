from pandas import read_csv, DataFrame
import pandas as pd
from dslabs_functions import get_variable_types
from sklearn.impute import SimpleImputer, KNNImputer
from pandas import concat
from numpy import ndarray

method_tag = "encoding"
file_tag = "class_ny_arrests"
filename = f"Data/{method_tag}/{file_tag}_{method_tag}.csv"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
print(f"Original data: {data.shape}")

#print columns
print(f"Columns: {data.columns}")

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
def get_first_values(data: DataFrame, value: int) -> DataFrame:
    firstValue = data[data_no_missing_values["PD_CD"] == value].head(1)
    #return firstValue[["KY_CD", "PD_DESC", "OFNS_DESC"]]
    return firstValue[["KY_CD"]]

# For all the missing values get the first KY_CD, PD_DESC, and OFNS_DESC values and substitute them
for index, row in missing_values.iterrows():
    first_values: DataFrame = get_first_values(data_no_missing_values, row["PD_CD"])
    if first_values.empty:
        continue
    missing_values.at[index, "KY_CD"] = first_values["KY_CD"].values[0]
    #missing_values.at[index, "PD_DESC"] = first_values["PD_DESC"].values[0]
    #missing_values.at[index, "OFNS_DESC"] = first_values["OFNS_DESC"].values[0]

missing_values.dropna(inplace=True)
print(f"Missing values recovered: {missing_values.shape}")

# Concatenate the data with missing values recovered with the data without missing values
data_no_missing_values = pd.concat([data_no_missing_values, missing_values])
print(f"Data after recovering missing values: {data_no_missing_values.shape}")

# Save the data with missing values recovered
data_no_missing_values.to_csv(f"./Data/{file_tag}_missing_values_method1.csv", index=True)


def mvi_by_filling(data: DataFrame, strategy: str = "frequent") -> DataFrame:
    df: DataFrame
    variables: dict = get_variable_types(data)
    stg_num, v_num = "mean", -1
    stg_sym, v_sym = "most_frequent", "NA"
    stg_bool, v_bool = "most_frequent", False
    if strategy != "knn":
        lst_dfs: list = []
        if strategy == "constant":
            stg_num, stg_sym, stg_bool = "constant", "constant", "constant"
        if len(variables["numeric"]) > 0:
            imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
            tmp_nr = DataFrame(
                imp.fit_transform(data[variables["numeric"]]),
                columns=variables["numeric"],
            )
            lst_dfs.append(tmp_nr)
        if len(variables["symbolic"]) > 0:
            imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
            tmp_sb = DataFrame(
                imp.fit_transform(data[variables["symbolic"]]),
                columns=variables["symbolic"],
            )
            lst_dfs.append(tmp_sb)
        if len(variables["binary"]) > 0:
            imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
            tmp_bool = DataFrame(
                imp.fit_transform(data[variables["binary"]]),
                columns=variables["binary"],
            )
            lst_dfs.append(tmp_bool)
        df = concat(lst_dfs, axis=1)
    else:
        imp = KNNImputer(n_neighbors=5)
        imp.fit(data)
        ar: ndarray = imp.transform(data)
        df = DataFrame(ar, columns=data.columns, index=data.index)
    return df

data_filled: DataFrame = mvi_by_filling(data_no_missing_values, "frequent")
print(f"Data after filling missing values: {data_filled.shape}")
data_filled.to_csv(f"./Data/{file_tag}_missing_values_method2.csv", index=True)

