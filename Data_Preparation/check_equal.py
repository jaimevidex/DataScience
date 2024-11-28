from pandas import read_csv, DataFrame

def get_unique_pd_cd_values(df):
    # Get all unique values from the PD_CD column
    unique_pd_cd_values = df['PD_CD'].unique()
    
    # Sort the values (optional, but often helpful)
    unique_pd_cd_values.sort()
    
    # Convert to a list (optional, for easier handling)
    unique_pd_cd_values = unique_pd_cd_values.tolist()
    del unique_pd_cd_values[-1]
    
    print(f"Number of unique PD_CD values: {len(unique_pd_cd_values)}")
    print("Unique PD_CD values:")
    print(unique_pd_cd_values)
    
    return unique_pd_cd_values

def check_consistent_values(df, group_column, check_column, group_value):
    # Filter the dataframe for the given group_column value
    filtered_df = df[df[group_column] == group_value]
    
    # Check if all check_column values are the same for the filtered records
    is_consistent = filtered_df[check_column].nunique() == 1
    
    if is_consistent:
        consistent_value = filtered_df[check_column].iloc[0]
        print(f"All records with {group_column} = {group_value} have the same {check_column} value: {consistent_value}")
    else:
        unique_values = filtered_df[check_column].unique()
        print(f"Records with {group_column} = {group_value} have different {check_column} values: {unique_values}")
    
    return is_consistent

filename = "Data/class_ny_arrests.csv"
file_tag = "class_ny_arrests"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
print(f"Original data: {data.shape}")

list_vals = get_unique_pd_cd_values(data)
for val in list_vals:
    check_consistent_values(data, "KY_CD", "OFNS_DESC", val)