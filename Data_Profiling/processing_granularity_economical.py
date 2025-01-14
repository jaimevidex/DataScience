from pandas import DataFrame, read_csv, Series
from numpy import arange, ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from dslabs_functions import plot_bar_chart, HEIGHT


def derive_time_granularities(df: DataFrame, column: str, levels: list[int]) -> DataFrame:
    """
    Create new columns for the given column where each new column
    represents the modulo operation with numbers from 1 to 14.
    """
    for level in levels:
        df[f"{column}_mod_{level}"] = df[column] % level
    return df


def analyse_time_granularity(data: DataFrame, column: str, levels: list[int]) -> ndarray:
    """
    Generate bar charts to analyze the distribution of remainders
    for the specified levels (1 to 14) on the Time column.
    """
    cols: int = len(levels)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {column}")

    for i, level in enumerate(levels):
        column_name = f"{column}_mod_{level}"
        counts: Series[int] = data[column_name].value_counts().sort_index()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=f"Modulo {level}",
            xlabel=f"mod {level}",
            ylabel="nr records",
            percentage=False,
        )
    return axs


# Main script
file_tag = "CLASS"
filename = "./Data/class_financial_distress.csv"  # Update with the correct path to your dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)

# The column containing numbers 1 to 14
time_column = "Time"

# Define granularity levels as numbers 1 to 14
granularity_levels = list(range(1, 15))

# Derive modulo granularities for the "Time" column
data_ext: DataFrame = derive_time_granularities(data, time_column, granularity_levels)

# Generate granularity analysis
analyse_time_granularity(data_ext, time_column, granularity_levels)
savefig(f"Data_Profiling/Images/Granularity_financial_distress/{file_tag}_granularity_{time_column}.png")
show()
