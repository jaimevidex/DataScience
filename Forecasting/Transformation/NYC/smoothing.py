from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure

file_tag = "NYC"
target = "Manhattan"
data: DataFrame = read_csv(
    "Data/forecast_ny_arrests_scaled.csv",
    index_col="Date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]

WIN_SIZE = 91
ss_smooth = series.rolling(window=WIN_SIZE).mean()

figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    ss_smooth.index.to_list(),
    ss_smooth.to_list(),
    xlabel=ss_smooth.index.name,
    ylabel=target,
    title=f"{file_tag} {target} after smoothing (size={WIN_SIZE})",
)
savefig(f"Forecasting/Images/Smoothing/{file_tag}_aftersmoothing.png")

sizes: list[int] = [30, 91, 183, 365]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
fig.suptitle(f"{file_tag} {target} after smoothing")

for i in range(len(sizes)):
    ss_smooth = data[target].rolling(window=sizes[i]).mean()
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),        
        ax=axs[i],
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"size={sizes[i]}",
    )
savefig(f"Forecasting/Images/Smoothing/{file_tag}_smoothingvarioussizes.png")

data = data.rolling(91).mean()
df : DataFrame = data

df.dropna(inplace=True)

#save series to csv
df.to_csv("Data/forecast_ny_arrests_smoothed.csv")
