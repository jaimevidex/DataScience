from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT
from pandas import Index, Period
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure

file_tag = "forecast_gdp_europe"
target = "GDP"
data: DataFrame = read_csv(
    "Data/Time_Series/forecast_gdp_europe_no_nan.csv",
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
    title=f"{file_tag} yearly {target}",
)
#show()

def ts_aggregation_by(
    data: Series | DataFrame,
    gran_level: str = "Y",
    agg_func: str = "mean",
) -> Series | DataFrame:
    df: Series | DataFrame = data.copy()
    index: Index[Period] = df.index.to_period(gran_level)
    df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
    df.index.drop_duplicates()
    df.index = df.index.to_timestamp()

    return df


ss_year: Series = ts_aggregation_by(series, "Y")
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    ss_year.index.to_list(),
    ss_year.to_list(),
    xlabel="year",
    ylabel=target,
    title=f"{file_tag} yearly mean {target}",
)
#show()
savefig(f"Forecasting/Images/Granularity/{file_tag}_univariate_aggregation_study.png")

study_tag = "granularity"
grans: list[str] = ["Y"]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(grans), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(grans)))
fig.suptitle(f" {target} aggregation study")

if len(grans) == 1:
    axs = [axs]
    
for i in range(len(grans)):
    ss: Series = ts_aggregation_by(series, grans[i])
    plot_line_chart(
        ss.index.to_list(),
        ss.to_list(),
        ax=axs[i],
        xlabel=f"{ss.index.name} ({grans[i]})",
        ylabel=target,
        #title=f"granularity={grans[i]}",
    )
#show()
savefig(f"Forecasting/Images/Granularity/{file_tag}_{study_tag}_study.png")
