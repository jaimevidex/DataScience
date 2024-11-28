from pandas import read_csv, DataFrame, Series
from dslabs_functions import get_variable_types, determine_outlier_thresholds_for_var


file_tag = "class_ny_arrests"
filename = f"Data/{file_tag}_missing_values.csv"
data: DataFrame = read_csv(filename, na_values="", index_col=None)
print(f"Original data: {data.shape}")

n_std: list [int] = [3,2,3,2,2,5,8,8,5]

numeric_vars: list[str] = get_variable_types(data)["numeric"]
if numeric_vars is not None:
    df: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data[numeric_vars].describe()
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=n_std[numeric_vars.index(var)]
        )
        outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    df.to_csv(f"./Data/{file_tag}_drop_outliers.csv", index=True)
    print(f"Data after dropping outliers: {df.shape}")
else:
    print("There are no numeric variables")