from pandas import Series, DataFrame, read_csv
from matplotlib.pyplot import subplots, show, gca, figure, show, savefig, plot, legend
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from dslabs_functions import HEIGHT, set_chart_labels, plot_line_chart, ts_aggregation_by

def plot_components(
    series: Series,
    title: str = "",
    x_label: str = "time",
    y_label: str = "",
) -> list[Axes]:
    decomposition: DecomposeResult = seasonal_decompose(series, model="add")
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(rows, 1, figsize=(12 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    return axs

image_tag = "stationarity"
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
ss_months: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)

plot_components(
    ss_months,
    title=f"{file_tag} monthly {target}",
    x_label=ss_months.index.name,
    y_label=target,
)

savefig(f"Forecasting/Images/Stationarity/{image_tag}_{file_tag}_components_{target}.png")

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_months.index.to_list(),
    ss_months.to_list(),
    xlabel=ss_months.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study",
    name="original",
)
n: int = len(ss_months)
plot(ss_months.index, [ss_months.mean()] * n, "r-", label="mean")
legend()

savefig(f"Forecasting/Images/Stationarity/{image_tag}_{file_tag}_stationary_{target}.png")

BINS = 10
mean_line: list[float] = []

for i in range(BINS):
    segment: Series = ss_months[i * n // BINS : (i + 1) * n // BINS]
    mean_value: list[float] = [segment.mean()] * (n // BINS)
    mean_line += mean_value
mean_line += [mean_line[-1]] * (n - len(mean_line))

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_months.index.to_list(),
    ss_months.to_list(),
    xlabel=ss_months.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study",
    name="original",
    show_stdev=True,
)
n: int = len(ss_months)
plot(ss_months.index, mean_line, "r-", label="mean")
legend()

savefig(f"Forecasting/Images/Stationarity/{image_tag}_{file_tag}_stationary_{target}_mean.png")

from statsmodels.tsa.stattools import adfuller


def eval_stationarity(series: Series) -> bool:
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05


print(f"The series {('is' if eval_stationarity(ss_months) else 'is not')} stationary")