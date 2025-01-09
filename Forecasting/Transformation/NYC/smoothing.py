from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT, mvi_by_filling
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure

file_tag = "NYC"
target = "Manhattan"
data: DataFrame = read_csv(
    "Data/train_forecasting_ny_arrests.csv",
    index_col="Date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
data2: DataFrame = read_csv(
    "Data/test_forecasting_ny_arrests.csv",
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

sizes: list[int] = [30, 91, 183, 365] # monthly, quarterly, semi-annual, and annual
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



smoothedData = data.rolling(91).mean()
df : DataFrame = smoothedData

mvi_by_filling(df, "knn")
df.dropna(inplace=True)

df.to_csv("Data/train_forecast_ny_arrests_smoothed.csv")