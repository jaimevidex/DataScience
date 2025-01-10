from numpy import sum, array
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show, subplots, savefig, setp
from dslabs_functions import HEIGHT, plot_line_chart, ts_aggregation_by, set_chart_labels, plot_multiline_chart
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
import numpy as np

image_tag = "distribution"
file_tag = "class_ny_arrests"
filename = "Data/forecast_ny_arrests.csv"
index = "Date"
target = "Manhattan"
data: DataFrame = read_csv(
    filename,
    index_col="Date",
    sep=";",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]
ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)

figure(figsize=(3*HEIGHT, HEIGHT))
plot_line_chart(
    ss_weeks.index.to_list(),
    ss_weeks.to_list(),
    xlabel="weeks",
    ylabel=target,
    title=f"{file_tag} weekly {target}",
)

savefig(f"Forecasting/Images/Distribution/{image_tag}_{file_tag}_weekly_{target}.png")

fig: Figure
axs: array
fig, axs = subplots(2, 2, figsize=(2 * HEIGHT, HEIGHT))
set_chart_labels(axs[0, 0], title="DAILY")
axs[0, 0].boxplot(series)
set_chart_labels(axs[0, 1], title="WEEKLY")
axs[0, 1].boxplot(ss_weeks)

axs[1, 0].grid(False)
axs[1, 0].set_axis_off()
axs[1, 0].text(0.2, 0, str(series.describe()), fontsize="small")

axs[1, 1].grid(False)
axs[1, 1].set_axis_off()
axs[1, 1].text(0.2, 0, str(ss_weeks.describe()), fontsize="small")

savefig(f"Forecasting/Images/Distribution/{image_tag}_{file_tag}_boxplot_{target}.png")

ss_days: Series = ts_aggregation_by(series, gran_level="D", agg_func=sum)
ss_months: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)
ss_quarters: Series = ts_aggregation_by(series, gran_level="Q", agg_func=sum)
ss_years: Series = ts_aggregation_by(series, gran_level="Y", agg_func=sum)

grans: list[Series] = [series, ss_months, ss_quarters, ss_years]
gran_names: list[str] = ["Daily", "Weekly", "Quarterly", "Annualy"]
fig: Figure
axs: array
fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
fig.suptitle(f"{file_tag} {target}")
for i in range(len(grans)):
    set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
    axs[i].hist(grans[i].values)

savefig(f"Forecasting/Images/Distribution/{image_tag}_{file_tag}_histograms_{target}.png")


def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series


figure(figsize=(3*HEIGHT, HEIGHT))
lags = get_lagged_series(ss_months, 20, 10)
plot_multiline_chart(ss_months.index.to_list(), lags, xlabel=index, ylabel=target)

savefig(f"Forecasting/Images/Distribution/{image_tag}_{file_tag}_lags_{target}.png")


def autocorrelation_study(series: Series, max_lag: int, delta: int = 1, bin_size: int = 25):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        shifted_values = series.shift(lag).tolist()
        
        # Perform binning
        data = list(zip(shifted_values, series_values))
        data = np.array(data)
        data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN values
        
        # Bin the data
        binned_data = []
        for j in range(0, len(data), bin_size):
            bin_slice = data[j:j + bin_size]
            if len(bin_slice) > 0:
                bin_mean = np.mean(bin_slice, axis=0)
                binned_data.append(bin_mean)
        binned_data = np.array(binned_data)
        
        # Scatter plot with binned data
        ax.scatter(binned_data[:, 0], binned_data[:, 1])
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
        
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    return


autocorrelation_study(series, 10, 1)

savefig(f"Forecasting/Images/Distribution/{image_tag}_{file_tag}_autocorrelation_{target}.png")