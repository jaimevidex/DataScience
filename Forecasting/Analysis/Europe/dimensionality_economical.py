from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
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
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} hourly {target}",
)
#show()
savefig(f"Forecasting/Images/Dimensionality/{file_tag}_univariate_study.png")

def plot_ts_multivariate_chart(data: DataFrame, title: str) -> list[Axes]:
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(data.shape[1], 1, figsize=(3 * HEIGHT, HEIGHT / 2 * data.shape[1]))
    fig.suptitle(title)

    for i in range(data.shape[1]):
        col: str = data.columns[i]
        plot_line_chart(
            data[col].index.to_list(),
            data[col].to_list(),
            ax=axs[i],
            xlabel=data.index.name,
            ylabel=col,
        )
    return axs

data: DataFrame = read_csv(
    "Data/Time_Series/forecast_gdp_europe_no_nan.csv",
    index_col="Year",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
print("Nr. Records = ", data.shape)
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])

plot_ts_multivariate_chart(data, title=f"{file_tag} {target}")
#show()
savefig(f"Forecasting/Images/Dimensionality/{file_tag}_multivariate_study.png")