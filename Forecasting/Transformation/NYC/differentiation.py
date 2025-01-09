from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig
from dslabs_functions import plot_line_chart, HEIGHT, mvi_by_filling

version: str = "_aggregated_M_91"
file_tag = "NYC"
filename = f"Data/train_forecast_ny_arrests_aggregated.csv"
index = "Date"
target = "Manhattan"
data: DataFrame = read_csv(
    filename,
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]
data2: DataFrame = read_csv(
    f"Data/test_forecast_ny_arrests_aggregated.csv",
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} daily {target}",
)

savefig(f"Forecasting/Images/Differentiation/{file_tag}_daily.png")

from dslabs_functions import plot_line_chart

ss_diff: Series = series.diff()
figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title="Differentiation",
    xlabel=series.index.name,
    ylabel=target,
)

savefig(f"Forecasting/Images/Differentiation/{file_tag}_diff.png")


from dslabs_functions import plot_ts_multivariate_chart

plot_ts_multivariate_chart(data, title=f"{file_tag} {target}")

savefig(f"Forecasting/Images/Differentiation/{file_tag}_multivariate.png")

diff_df_train: DataFrame = data.diff()
diff_df_test: DataFrame = data2.diff()

plot_ts_multivariate_chart(diff_df_train, title=f"{file_tag} {target} - after first differentiation")

savefig(f"Forecasting/Images/Differentiation/{file_tag}_after_first_diff.png")

mvi_by_filling(diff_df_train, "knn")
mvi_by_filling(diff_df_test, "knn")
diff_df_train.dropna(inplace=True)
diff_df_test.dropna(inplace=True)


diff_df_train.to_csv("Data/train_forecast_ny_arrests_differentiated.csv")
diff_df_test.to_csv("Data/test_forecast_ny_arrests_differentiated.csv")

ss_diff2: DataFrame = ss_diff.diff()

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff2.index.to_list(),
    ss_diff2.to_list(),
    title="Differentiation",
    xlabel=series.index.name,
    ylabel=target,
)

savefig(f"Forecasting/Images/Differentiation/{file_tag}_diff2.png")

plot_ts_multivariate_chart(data, title=f"{file_tag} {target}")

savefig(f"Forecasting/Images/Differentiation/{file_tag}_multivariate2.png")

diff_df: DataFrame = data.diff().diff()

plot_ts_multivariate_chart(diff_df, title=f"{file_tag} {target} - after second differentiation")

savefig(f"Forecasting/Images/Differentiation/{file_tag}_after_second_diff.png")

