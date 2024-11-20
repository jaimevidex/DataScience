""" Dimensionality """
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart, get_variable_types

# Load the data and print the number of variables and records
filename = "./Security/class_ny_arrests.csv"
file_tag = "class_ny_arrests"
data: DataFrame = read_csv(filename, na_values="", index_col=None)

print(data.shape)

# Plot a graph with each of them
figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"Images/{file_tag}_records_variables.png")
#show()

# Missing Values of each variable
mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"Images/{file_tag}_mv.png")
#show()

# Show the different Variable Types

variable_types: dict[str, list] = get_variable_types(data)
print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"Images/{file_tag}_variable_types.png")
show()