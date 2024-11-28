from pandas import read_csv, DataFrame, Series
from dslabs_functions import get_variable_types, determine_outlier_thresholds_for_var
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, subplots, show, figure
from dslabs_functions import define_grid, HEIGHT, plot_multibar_chart
from numpy import ndarray, log



filename = "Data/class_ny_arrests_missing_values.csv"
file_tag = "class_ny_arrests"
data: DataFrame = read_csv(filename, na_values="", index_col=None)
print(f"Original data: {data.shape}")

NR_STDEV: list [int] = [3,2,3,2,2,5,8,8,5,2,2,2]
IQR_FACTOR: float = [1.5,1.5,2.0,1.5,0.0,3.7,8.5,8.5,3.6,1.5,1.5, 1.5]

numeric_vars: list[str] = get_variable_types(data)["numeric"]

print("Boxplots")
if [] != numeric_vars:
    rows: int
    cols: int
    rows, cols = define_grid(len(numeric_vars))
    fig: Figure
    axs: ndarray
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title("Boxplot for %s" % numeric_vars[n])
        axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"Data_Preparation/Images/{file_tag}_single_boxplots.png")
    #show()
else:
    print("There are no numeric variables.")


def determine_outlier_thresholds_for_var(summary5: Series, std_based: bool = True, threshold: float = NR_STDEV) -> tuple[float, float]:
    top: float = 0
    bottom: float = 0
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom

# Function to count the number of outliers for each variable

def count_outliers(
    data: DataFrame,
    numeric: list[str],
    nrstdev: list[int] = NR_STDEV,
    iqrfactor: list[float] = IQR_FACTOR,
) -> dict:
    outliers_iqr: list = []
    outliers_stdev: list = []
    summary5: DataFrame = data[numeric].describe()

    for var in numeric:
        top: float
        bottom: float
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev[numeric.index(var)]
        )
        outliers_stdev += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor[numeric.index(var)]
        )
        outliers_iqr += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}

# Show the number of outliers for each variable
print("Outliers")
if [] != numeric_vars:
    outliers: dict[str, int] = count_outliers(data, numeric_vars)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric_vars,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"Data_Preparation/Images/{file_tag}_outliers_standard.png")
    #show()
else:
    print("There are no numeric variables.")


if numeric_vars is not None:
    df: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data[numeric_vars].describe()
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=NR_STDEV[numeric_vars.index(var)]
        )
        outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    df.to_csv(f"./Data/{file_tag}_drop_outliers.csv", index=True)
    print(f"Data after dropping outliers: {df.shape}")
else:
    print("There are no numeric variables")