from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib.pyplot import subplots, show, savefig
file = "class_financial_distress"
filename = "./Data/class_financial_distress.csv"
data: DataFrame = read_csv(filename, index_col=None, na_values="")
target = "CLASS"
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    data
)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
df_zscore.to_csv(f"Data/{file}_scaled_zscore.csv", index=None)


transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"Data/{file}_scaled_minmax.csv", index=None)

from matplotlib.pyplot import subplots, show

fig, axs = subplots(1, 1, figsize=(100, 10), squeeze=False)

axs[0, 0].set_title("Z-score normalization")
data.boxplot(ax=axs[0, 0])
axs[0, 0].tick_params(axis='x', rotation=-45)
show()

axs[0, 0].set_title("Original data")
df_zscore.boxplot(ax=axs[0, 0])
axs[0, 0].tick_params(axis='x', rotation=-45)
show()

axs[0, 0].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 0])
axs[0, 0].tick_params(axis='x', rotation=-45)
show()




