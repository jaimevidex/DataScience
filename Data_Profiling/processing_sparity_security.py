from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart, plot_scatter_chart
from seaborn import heatmap
from dslabs_functions import get_variable_types

filename = "./Data/Previous/class_ny_arrests.csv"
file_tag = "class_ny_arrests"
data: DataFrame = read_csv(filename, na_values="", index_col=None)
data = data.dropna()


vars: list = data.columns.to_list()
"""
if [] != vars:
    target = "CLASS"

    n: int = len(vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        print(f"Processing {i + 1} of {len(vars)}")
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            print(f"Processing {i + 1}: {j + 1} of {len(vars)}")
            var2: str = vars[j]
            plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j - 1])
    savefig(f"Data_Profiling/Images/{file_tag}_sparsity_study.png")
    show()
else:
    print("Sparsity class: there are no variables.")




# Sparsity per class individual
if [] != vars:
    target = "CLASS"


    for i in range(len(vars)):
        print(f"Processing {i + 1} of {len(vars)}")
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            fig, ax = subplots(figsize=(HEIGHT, HEIGHT))
            var2: str = vars[j]
            plot_multi_scatters_chart(data,var1, var2, target,ax=ax)
            print(f"Saving {file_tag}_{i+1}_{j+1}_sparsity_per_class_study.png")
            savefig(f"Data_Profiling/Images/153/{file_tag}_{i+1}_{j+1}_sparsity_per_class_study.png")
    else:
        print("Sparsity per class: there are no variables.")
"""

# Sparsity per class all
if [] != vars:
    target = "CLASS"

    n: int = len(vars) - 1
    fig, axs = subplots(n, n, figsize=(n * HEIGHT/2, n * HEIGHT/2), squeeze=False)
    for i in range(len(vars)):
        print(f"Processing {i + 1} of {len(vars)}")
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
            print(f"Processing {i + 1}: {j + 1} of {len(vars)}")
    savefig(f"Data_Profiling/Images/{file_tag}_sparsity_per_class_study.png")
    #show()
else:
    print("Sparsity per class: there are no variables.")

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
corr_mtx: DataFrame = data[numeric].corr().abs()

figure(figsize=(9, 6))
heatmap(
    abs(corr_mtx),
    xticklabels=numeric,
    yticklabels=numeric,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"Data_Profiling/Images/{file_tag}_correlation_analysis.png")
#show()
