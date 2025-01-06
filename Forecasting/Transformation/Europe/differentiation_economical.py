from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT
from dslabs_functions import plot_line_chart
from dslabs_functions import plot_ts_multivariate_chart


file_tag = "forecast_gdp_europe"
filename = "Data/Time_Series/forecast_gdp_europe_aggregated.csv"
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

figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} hourly {target}",
)
#show()
study_tag = "hourly_meter_reading"
savefig(f"Forecasting/Images/Differentiation/{file_tag}_{study_tag}_study.png")

ss_diff: Series = series.diff()
figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title="Differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
#show()
study_tag = "differentiation"
savefig(f"Forecasting/Images/Differentiation/{file_tag}_{study_tag}_study.png")

plot_ts_multivariate_chart(data, title=f"{file_tag} {target}")
#show()
study_tag = "Multivariate_Time_Series"
savefig(f"Forecasting/Images/Differentiation/{file_tag}_{study_tag}_study.png")

diff_df: DataFrame = data.diff()

plot_ts_multivariate_chart(diff_df, title=f"{file_tag} {target} - after first differentiation")
#show()
study_tag = "first_differentiation"
savefig(f"Forecasting/Images/Differentiation/{file_tag}_{study_tag}_study.png")


diff_df: DataFrame = diff_df.diff()

plot_ts_multivariate_chart(diff_df, title=f"{file_tag} {target} - after second differentiation")
#show()
study_tag = "second_differentiation"
savefig(f"Forecasting/Images/Differentiation/{file_tag}_{study_tag}_study.png")