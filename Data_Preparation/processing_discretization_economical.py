from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types
from sklearn.preprocessing import KBinsDiscretizer
from pandas import concat
from matplotlib.pyplot import subplots, show, savefig

filename = "./Data/class_financial_distress_smote.csv"
file = "class_financial_distress"

data: DataFrame = read_csv(
    filename,
    index_col=None,
    na_values="",
    parse_dates=True,
)

variable_types: dict[str, list] = get_variable_types(data)
numeric_vars: list[str] = variable_types["numeric"]
symbolic_vars: list[str] = variable_types["symbolic"]
boolean_vars: list[str] = variable_types["binary"]

df_nr: DataFrame = data[numeric_vars]
df_sb: DataFrame = data[symbolic_vars]
df_bool = data[boolean_vars]

N_BINS = 5
discretization: KBinsDiscretizer = KBinsDiscretizer(
    n_bins=N_BINS, encode="ordinal", strategy="uniform"
)
discretization.fit(df_nr)
eq_width = DataFrame(discretization.transform(df_nr), index=data.index)

df = DataFrame(df_sb, index=data.index)
df: DataFrame = concat([df, df_bool, eq_width], axis=1)
df.columns = symbolic_vars + boolean_vars + numeric_vars
df.to_csv(f"data/{file}_eq_width.csv", index=True)

df.hist(bins=N_BINS)
"""show()"""

discretization: KBinsDiscretizer = KBinsDiscretizer(
    n_bins=N_BINS, encode="ordinal", strategy="quantile"
)
discretization.fit(df_nr)
eq_width = DataFrame(discretization.transform(df_nr), index=data.index)

df = DataFrame(df_sb, index=data.index)
df = concat([df, df_bool, eq_width], axis=1)
df.columns = symbolic_vars + boolean_vars + numeric_vars
df.to_csv(f"data/{file}_eq_frequency.csv", index=True)

df.hist(bins=N_BINS)
"""show()"""