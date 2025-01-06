from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT
from dslabs_functions import plot_ts_multivariate_chart


file_tag = "forecast_gdp_europe"
target = "GDP"
data: DataFrame = read_csv(
    "Data/Time_Series/forecast_gdp_europe_smoothing.csv",
    index_col="Year",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

series: Series = data[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} hourly {target}",
)
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
ss_agg: Series = ts_aggregation_by(series, gran_level="D", agg_func="sum")
plot_line_chart(
    ss_agg.index.to_list(),
    ss_agg.to_list(),
    xlabel=ss_agg.index.name,
    ylabel=target,
    title=f"{file_tag} daily {target}",
)
#show()
study_tag = "before"
savefig(f"Forecasting/Images/Aggregation/{file_tag}_{study_tag}_study.png")


plot_ts_multivariate_chart(data, title=f"{file_tag} {target}")

agg_df: DataFrame = ts_aggregation_by(data, gran_level="W", agg_func="mean")

plot_ts_multivariate_chart(agg_df, title=f"{file_tag} after hourly aggregation {target}")
#show()
study_tag = "after"
savefig(f"Forecasting/Images/Aggregation/{file_tag}_{study_tag}_study.png")

output_file = "Data/Time_Series/forecast_gdp_europe_aggregated.csv"
agg_df.to_csv(output_file, sep=",", decimal=".", index=True)