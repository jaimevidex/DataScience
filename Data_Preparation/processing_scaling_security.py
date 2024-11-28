"""
Standard Scaler (Z-score) for the NY Arrests dataset
"""
from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import savefig, subplots, show

file_tag = "class_ny_arrests"
filename = f"Data/{file_tag}_drop_outliers.csv"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
print(f"Original data: {data.shape}")
target = "LAW_CAT_CD"
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)

#transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
#df_zscore = DataFrame(transf.transform(data), columns=data.columns, index=data.index)
#df_zscore[target] = target_data
#df_zscore.to_csv(f"Data/{file_tag}_scaled_zscore.csv", index=None)
"""
MinMax Scaler for the NY Arrests dataset
"""
transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), columns=data.columns, index=data.index)
df_minmax[target] = target_data
df_minmax.to_csv(f"Data/{file_tag}_scaled_minmax.csv", index=None)

"""
Results
"""
file_tag = "class_ny_arrests"
filename = f"Data/{file_tag}_scaled_minmax.csv"
df_minmax: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
filename = f"Data/{file_tag}_scaled_zscore.csv"
df_zscore: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")

fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 1].set_title("Original data")
data.boxplot(ax=axs[0, 0])
axs[0, 0].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
#show()
savefig(f"./{file_tag}_comparison.png")