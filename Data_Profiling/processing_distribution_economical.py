""" Dimensionality """
from pandas import read_csv, DataFrame, Series

from matplotlib.pyplot import figure, savefig, show, savefig, show, subplots, ylim
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dslabs_functions import plot_bar_chart, get_variable_types, define_grid, HEIGHT, plot_multibar_chart, set_chart_labels, plot_multiline_chart

from numpy import ndarray, log

from scipy.stats import norm, expon, lognorm


# Load the data and print the number of variables and records
filename = "./Data/class_financial_distress.csv"
file_tag = "class_financial_distress"
data: DataFrame = read_csv(filename, na_values="", index_col=None)

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
"""
# Boxplot for all numeric variables
print("General Boxplot")
if [] != numeric:
    figure(figsize=(10, 9))
    data[numeric].boxplot(rot=45)
    ylim(-300000, 350000)  # Set the y-axis limits as needed
    savefig(f"Images/{file_tag}_global_boxplot.png")
    #plt.show()
else:
    print("There are no numeric variables.")
"""
# Boxplot for each numeric variable
print("Boxplots")
if [] != numeric:
    rows: int
    cols: int
    rows, cols = define_grid(len(numeric))
    fig: Figure
    axs: ndarray
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        axs[i, j].set_title("Boxplot for %s" % numeric[n])
        axs[i, j].boxplot(data[numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"Images/{file_tag}_single_boxplots.png")
    #show()
else:
    print("There are no numeric variables.")


NR_STDEV: float = [2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 6, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   2, 2, 2, 
                   6, 2, 2, 
                   6, 2, 6, 
                   2, 2, 1, 
                   2, 2, 2, 
                   2, 2, 8, 
                   2, 2, 2, 
                   0.5, 2, 2, 
                   2, 2]
IQR_FACTOR: float = [1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,0.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1,1,
                     1,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,0.5,
                     0.5,0.5,0.5,
                     1,0,5,1,
                     1.5,1.5,0.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1.5,1.5,1.5,
                     1,1.5,1.5,
                     1.5,1.5]


# Function to determine the outlier thresholds for a variable

def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
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

# Show the box plots without outliers
print("Boxplots without outliers")
if [] != numeric:
    rows, cols = define_grid(len(numeric))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        top, bottom = determine_outlier_thresholds_for_var(
            data[numeric[n]].describe(), std_based=False, threshold=IQR_FACTOR[numeric.index(numeric[n])]
        )
        data_no_outliers: DataFrame = data[
            (data[numeric[n]] >= bottom) & (data[numeric[n]] <= top)
        ]
        axs[i, j].set_title("Boxplot for %s without outliers" % numeric[n])
        axs[i, j].boxplot(data_no_outliers[numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"Images/{file_tag}_single_boxplots_no_outliers.png")
    #show()
else:
    print("There are no numeric variables.")

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
if [] != numeric:
    outliers: dict[str, int] = count_outliers(data, numeric)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"Images/{file_tag}_outliers_standard.png")
    #show()
else:
    print("There are no numeric variables.")

"""
# Histogram for each numeric variable
print("Histograms")
if [] != numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i: int
    j: int
    i, j = 0, 0
    for n in range(len(numeric)):
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(data[numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"Images/{file_tag}_single_histograms_numeric.png")
    #show()
else:
    print("There are no numeric variables.")

# Compute distributions for a list of values

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions["LogNor(%.1f,%.2f)" % (log(scale), sigma)] = lognorm.pdf(
        x_values, sigma, loc, scale
    )
    return distributions

# Histogram with distributions for each numeric variable

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values: list = series.sort_values().to_list()
    ax.hist(values, 20, density=True)
    distributions: dict = compute_known_distributions(values)
    plot_multiline_chart(
        values,
        distributions,
        ax=ax,
        title="Best fit for %s" % var,
        xlabel=var,
        ylabel="",
    )

print("Histograms with distributions")
if [] != numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        histogram_with_distributions(axs[i, j], data[numeric[n]].dropna(), numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"Images/{file_tag}_histogram_numeric_distribution.png")
    #show()
else:
    print("There are no numeric variables.")



symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
print(symbolic)
if [] != symbolic:
    rows, cols = define_grid(len(symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(symbolic)):
        counts: Series = data[symbolic[n]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % symbolic[n],
            xlabel=symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"Images/{file_tag}_histograms_symbolic.png")
    #show()
else:
    print("There are no symbolic variables.")


target = "CLASS"
print(data[target].value_counts())
values: Series = data[target].value_counts()
print(values)

figure(figsize=(4, 2))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)
savefig(f"Images/{file_tag}_class_distribution.png")"""