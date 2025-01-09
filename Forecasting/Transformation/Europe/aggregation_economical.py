"""from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT
from dslabs_functions import plot_ts_multivariate_chart


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
    title=f"{file_tag} yearly {target}",
)
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
ss_agg: Series = ts_aggregation_by(series, gran_level="Y", agg_func="sum")
plot_line_chart(
    ss_agg.index.to_list(),
    ss_agg.to_list(),
    xlabel=ss_agg.index.name,
    ylabel=target,
    title=f"{file_tag} yearly {target}",
)
#show()
study_tag = "before"
savefig(f"Forecasting/Images/Aggregation/{file_tag}_{study_tag}_study.png")

plot_ts_multivariate_chart(data, title=f"{file_tag} {target}")

agg_df: DataFrame = ts_aggregation_by(data, gran_level="Y", agg_func="mean")

plot_ts_multivariate_chart(agg_df, title=f"{file_tag} after yearly aggregation {target}")
#show()
study_tag = "after"
savefig(f"Forecasting/Images/Aggregation/{file_tag}_{study_tag}_study.png")

output_file = "Data/Time_Series/forecast_gdp_europe_aggregated.csv"
agg_df.to_csv(output_file, sep=",", decimal=".", index=True)

train, test = series_train_test_split(agg_df, trn_pct=0.90)
test.to_csv("Data/Time_Series/forecast_gdp_europe_aggregation_year_test.csv", index=True)
train.to_csv("Data/Time_Series/forecast_gdp_europe_aggregation_year_train.csv", index=True)
"""

#for decade
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig
from dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT
from dslabs_functions import plot_ts_multivariate_chart


file_tag = "forecast_gdp_europe_decade"
target = "GDP"
data: DataFrame = read_csv(
    "Data/Time_Series/forecast_gdp_europe_no_nan.csv",
    index_col="Year",
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

# Create a Decade column for aggregation

data["Decade"] = (data.index.year // 10) * 10  # Create a column for the decade

# Plot original series
series: Series = data[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} decade {target}",
)
study_tag = "before_decade"
savefig(f"Forecasting/Images/Aggregation/{file_tag}_{study_tag}_study.png")


# Aggregate by decade (mean GDP per decade)

agg_func: str = "mean"
agg_df: DataFrame = data.groupby("Decade").agg(agg_func)

# Plot the aggregated data
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    agg_df.index.to_list(),
    agg_df[target].to_list(),
    xlabel="Decade",
    ylabel=target,
    title=f"{file_tag} aggregated by decade",
)
study_tag = "after_decade"
savefig(f"Forecasting/Images/Aggregation/{file_tag}_{study_tag}_study.png")

# Multivariate plot for aggregated data
plot_ts_multivariate_chart(agg_df, title=f"{file_tag} after decade aggregation {target}")
savefig(f"Forecasting/Images/Aggregation/{file_tag}_multivariate_after_decade.png")

# Save the aggregated data to a CSV file
output_file = "Data/Time_Series/forecast_gdp_europe_aggregated_decades.csv"
agg_df.to_csv(output_file, sep=",", decimal=".", index=True)

# Train-test split for decade-aggregated data
train, test = series_train_test_split(agg_df, trn_pct=0.90)

# Save train and test datasets
test.to_csv("Data/Time_Series/forecast_gdp_europe_aggregation_decade_test.csv", index=True)
train.to_csv("Data/Time_Series/forecast_gdp_europe_aggregation_decade_train.csv", index=True)
