from pandas import read_csv, DataFrame, Series
from dslabs_functions import get_variable_types, determine_outlier_thresholds_for_var


file_tag = "class_financial_distress"
filename = f"Data/{file_tag}.csv"
data: DataFrame = read_csv(filename, na_values="", index_col=None)
print(f"Original data: {data.shape}")

n_std: list [int] = [2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 6, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   6, 2, 2, 
                   6, 2, 6, 
                   2, 2, 1, 
                   2, 2, 2, 
                   2, 2, 8, 
                   2, 2, 2, 
                   1, 2, 2, 
                   2, 2]

numeric_vars: list[str] = get_variable_types(data)["numeric"]
filtered_numeric_vars = [col for col in numeric_vars if col not in ["Time", "Company"]]

print("Dropping Outliers")
if filtered_numeric_vars is not None:
    df: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data[filtered_numeric_vars].describe()
    for var in filtered_numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=n_std[filtered_numeric_vars.index(var)]
        )
        outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    df.to_csv(f"./Data/{file_tag}_drop_outliers.csv", index=True)
    print(f"Data after dropping outliers: {df.shape}")
else:
    print("There are no numeric variables")

print("Truncating Outliers")
if [] != filtered_numeric_vars:
    df: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data[filtered_numeric_vars].describe()
    for var in filtered_numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        df[var] = df[var].apply(
            lambda x: top if x > top else bottom if x < bottom else x
        )
    df.to_csv(f"data/{file_tag}_truncate_outliers.csv", index=True)
    print("Data after truncating outliers:", df.shape)
else:
    print("There are no numeric variables")

print("Replacing Outliers")
if [] != filtered_numeric_vars:
    df: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data[filtered_numeric_vars].describe()
    for var in filtered_numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        median: float = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
    df.to_csv(f"data/{file_tag}_replacing_outliers.csv", index=True)
    print("Data after replacing outliers:", df.shape)
else:
    print("There are no numeric variables")
