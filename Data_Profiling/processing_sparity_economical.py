from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots, savefig, show
from seaborn import heatmap
from dslabs_functions import HEIGHT, plot_multi_scatters_chart
from dslabs_functions import get_variable_types

# Constants
HEIGHT = 2  # Height multiplier for figure size
file_tag = "class_financial_distress"
filename = "./Data/class_financial_distress.csv"

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)
data = data.dropna()

vars: list = data.columns.to_list()

if vars:
    target = "CLASS"
    n: int = len(vars) - 1

    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j - 1])

    #savefig(f"images/{file_tag}_sparsity_study.png")
    #show()
else:
    print("Sparsity class: there are no variables.")


variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
corr_mtx: DataFrame = data[numeric].corr().abs()

# Plot the correlation heatmap
figure()
heatmap(
    abs(corr_mtx),
    xticklabels=numeric,
    yticklabels=numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)

high_corr_pairs = [
    (var1, var2, corr_mtx.loc[var1, var2])
    for i, var1 in enumerate(numeric)
    for var2 in numeric[i + 1:]
    if corr_mtx.loc[var1, var2] > 0.9
]

print("Highly correlated variable pairs (correlation > 0.9):")
for var1, var2, corr in high_corr_pairs:
    print(f"{var1} - {var2}: {corr:.2f}")
#savefig(f"Data_Profiling/Images/Sparsity_financial_distress/{file_tag}_correlation_analysis_testing.png")
#show()

