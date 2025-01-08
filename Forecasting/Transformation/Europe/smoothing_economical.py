from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, HEIGHT
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

def series_train_test_split(data: Series, trn_pct: float = 0.90) -> tuple[Series, Series]:
    trn_size: int = int(len(data) * trn_pct)
    df_cp: Series = data.copy()
    train: Series = df_cp.iloc[:trn_size]
    test: Series = df_cp.iloc[trn_size:]
    return train, test


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


WIN_SIZE = 10
ss_smooth: Series = series.rolling(window=WIN_SIZE).mean()

figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    ss_smooth.index.to_list(),
    ss_smooth.to_list(),
    xlabel=ss_smooth.index.name,
    ylabel=target,
    title=f"{file_tag} {target} after smoothing (size={WIN_SIZE})",
)
#show()
study_tag = "smoothing1"
savefig(f"Forecasting/Images/Smoothing/{file_tag}_{study_tag}_study10.png")

sizes: list[int] = [5, 10]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
fig.suptitle(f"{file_tag} {target} after smoothing")

for i in range(len(sizes)):
    ss_smooth: Series = series.rolling(window=sizes[i]).mean()
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),
        ax=axs[i],
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"size={sizes[i]}",
    )
#show()
study_tag = "smoothing2"
savefig(f"Forecasting/Images/Smoothing/{file_tag}_{study_tag}_study.png")


train, test = series_train_test_split(data, trn_pct=0.90)
test.to_csv("Data/Time_Series/forecast_gdp_europe_smoothing_test.csv", index=True)

ss_train: DataFrame = train.rolling(10).mean()
ss_train.to_csv("Data/Time_Series/forecast_gdp_europe_smoothing10_train.csv", index=True)

ss_smooth: DataFrame = data.rolling(10).mean()
ss_smooth.to_csv("Data/Time_Series/forecast_gdp_europe_smoothing10.csv")