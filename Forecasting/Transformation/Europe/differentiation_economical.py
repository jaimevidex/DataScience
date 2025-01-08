from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT
from dslabs_functions import plot_line_chart
from dslabs_functions import plot_ts_multivariate_chart


file_tag = "forecast_gdp_europe"
filename = "Data/Time_Series/forecast_gdp_europe_no_nan.csv"
index = "Year"
target = "GDP"
data: DataFrame = read_csv(
    filename,
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]

def series_train_test_split(data: Series, trn_pct: float = 0.90) -> tuple[Series, Series]:
    trn_size: int = int(len(data) * trn_pct)
    df_cp: Series = data.copy()
    train: Series = df_cp.iloc[:trn_size]
    test: Series = df_cp.iloc[trn_size:]
    return train, test

figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} yearly {target}",
)

savefig(f"Forecasting/Images/Differentiation/{file_tag}_yearly.png")

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

diff_df: DataFrame = data.diff()

train, test = series_train_test_split(diff_df, trn_pct=0.90)
test.to_csv("Data/Time_Series/forecast_gdp_europe_first_diff_test.csv", index=True)
train.to_csv("Data/Time_Series/forecast_gdp_europe_first_diff_train.csv", index=True)

plot_ts_multivariate_chart(diff_df, title=f"{file_tag} {target} - after first differentiation")

savefig(f"Forecasting/Images/Differentiation/{file_tag}_after_first_diff.png")

ss_diff2: Series = ss_diff.diff()

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

train, test = series_train_test_split(diff_df, trn_pct=0.90)
test.to_csv("Data/Time_Series/forecast_gdp_europe_second_diff_test.csv", index=True)
train.to_csv("Data/Time_Series/forecast_gdp_europe_second_diff_train.csv", index=True)

plot_ts_multivariate_chart(diff_df, title=f"{file_tag} {target} - after second differentiation")

savefig(f"Forecasting/Images/Differentiation/{file_tag}_after_second_diff.png")

