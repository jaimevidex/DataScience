from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig
from dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT

file_tag = "NYC"
target = "Manhattan"
data: DataFrame = read_csv(
    f"Data/train_forecast_ny_arrests_smoothed.csv",
    index_col="Date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
data2: DataFrame = read_csv(
    f"Data/test_forecasting_ny_arrests.csv",
    index_col="Date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

series: Series = data[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
ss_agg: Series = ts_aggregation_by(series, gran_level="W", agg_func="sum")
plot_line_chart(
    ss_agg.index.to_list(),
    ss_agg.to_list(),
    xlabel=ss_agg.index.name,
    ylabel=target,
    title=f"{file_tag} weekly {target}",
)

savefig(f"Forecasting/Images/Aggregation/{file_tag}_W_aggregation.png")

from dslabs_functions import plot_ts_multivariate_chart

plot_ts_multivariate_chart(data, title=f"{file_tag} {target}")

#agg_df: DataFrame = ts_aggregation_by(data, gran_level="D", agg_func="mean")
#agg_df2: DataFrame = ts_aggregation_by(data2, gran_level="D", agg_func="mean")

#plot_ts_multivariate_chart(agg_df, title=f"{file_tag} after daily aggregation {target}")

agg_df: DataFrame = ts_aggregation_by(data, gran_level="Y", agg_func="mean")
agg_df2: DataFrame = ts_aggregation_by(data2, gran_level="Y", agg_func="mean")

agg_df.to_csv(f"Data/train_forecast_ny_arrests_aggregated.csv")
agg_df2.to_csv(f"Data/test_forecast_ny_arrests_aggregated.csv")


