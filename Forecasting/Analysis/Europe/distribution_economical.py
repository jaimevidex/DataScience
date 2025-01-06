from numpy import sum
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by
from numpy import array
from matplotlib.pyplot import show, subplots
from matplotlib.figure import Figure
from dslabs_functions import set_chart_labels
from dslabs_functions import plot_multiline_chart
from matplotlib.pyplot import setp
from matplotlib.gridspec import GridSpec


file_tag = "forecast_gdp_europe"
filename = "Data/Time_Series/forecast_gdp_europe.csv"
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
ss_annual: Series = ts_aggregation_by(series, gran_level="Y", agg_func=sum)

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_annual.index.to_list(),
    ss_annual.to_list(),
    xlabel="Year",
    ylabel=target,
    title=f"{file_tag} Annual {target}",
)
#show()
study_tag = "time_series_aggregation_by"
savefig(f"Forecasting/Images/{file_tag}_{study_tag}_study.png")

fig: Figure
axs: array
fig, axs = subplots(2, 1, figsize=(HEIGHT, 2 * HEIGHT))

# Boxplot on top
set_chart_labels(axs[0], title="ANNUAL")
axs[0].boxplot(ss_annual)

# 5-Number Summary below
axs[1].grid(False)
axs[1].set_axis_off()

# Add descriptive statistics as text
summary_stats = str(ss_annual.describe())
axs[1].text(0.5, 0.5, summary_stats, fontsize="small", ha="center", va="center", wrap=True)

# Save the combined figure
study_tag = "5-Number_Summary"
savefig(f"Forecasting/Images/{file_tag}_{study_tag}_study.png")


grans: list[Series] = [ss_annual]
gran_names: list[str] = ["Annualy"]
fig: Figure
axs: array
fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
fig.suptitle(f"{file_tag} {target}")
if len(grans) == 1:
    axs = [axs]

for i in range(len(grans)):
    set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
    axs[i].hist(grans[i].values)
#show()
study_tag = "variables_distribution"
savefig(f"Forecasting/Images/{file_tag}_{study_tag}_study.png")

def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series


figure(figsize=(3 * HEIGHT, HEIGHT))
lags = get_lagged_series(series, 20, 10)
plot_multiline_chart(series.index.to_list(), lags, xlabel=index, ylabel=target)
study_tag = "lagged_series"
savefig(f"Forecasting/Images/{file_tag}_{study_tag}_study.png")

def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    return


autocorrelation_study(series, 10, 1)
study_tag = "autocorrelation"
savefig(f"Forecasting/Images/{file_tag}_{study_tag}_study.png")