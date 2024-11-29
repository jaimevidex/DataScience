from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types
from sklearn.preprocessing import LabelEncoder
from math import pi, sin, cos
from numpy import ndarray
from pandas import concat
from sklearn.preprocessing import OneHotEncoder

filename = "Data/class_ny_arrests.csv"
file_tag = "class_ny_arrests"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")
print(f"Original data: {data.shape}")

vars: dict[str, list] = get_variable_types(data)

"""
Binary encoding
"""
felony_misdemeanor: dict[str, int] = {"F": 0, "M": 1}
male_female: dict[str, int] = {"M": 0, "F": 1}

"""
Nominal encoding
"""
'''
perp_race: dict[str, int] = {
    "UNKNOWN": 0,
    "WHITE": 1,
    "WHITE HISPANIC": 2,
    "AMERICAN INDIAN/ALASKAN NATIVE": 3,
    "ASIAN / PACIFIC ISLANDER": 4,
    "BLACK HISPANIC": 5,
    "BLACK": 6,
    "OTHER": 7
}
'''

def dummify(df: DataFrame, vars_to_dummify: list[str]) -> DataFrame:
    other_vars: list[str] = [c for c in df.columns if not c in vars_to_dummify]

    enc = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, drop="if_binary"
    )
    trans: ndarray = enc.fit_transform(df[vars_to_dummify])

    new_vars: ndarray = enc.get_feature_names_out(vars_to_dummify)
    dummy = DataFrame(trans, columns=new_vars, index=df.index)

    final_df: DataFrame = concat([df[other_vars], dummy], axis=1)
    return final_df


data = dummify(data, ["PERP_RACE"])


arrest_boro: dict[str, int] = {
    "B": 0,
    "K": 1,
    "M": 2,
    "Q": 3,
    "S": 4
}

age_group: dict[str, int] = {
    "UNKNOWN": -1,
    "<18": 0,
    "18-24": 1,
    "25-44": 2,
    "45-64": 3,
    "65+": 4
}

monthly_val:dict[int, float] = {
    1: 0,
    2: pi/6,
    3: pi/3,
    4: pi/2,
    5: 2*pi/3,
    6: 5*pi/6,
    7: pi,
    8: -5*pi/6,
    9: -2*pi/3,
    10: -pi/2,
    11: -pi/3,
    12: -pi/6
}

def encode_cyclic_variables(data: DataFrame, vars: list[str]) -> None:
    for v in vars:
        x_max: float | int = max(data[v])
        data[v + "_sin"] = data[v].apply(lambda x: round(sin(2 * pi * x / x_max), 3))
        data[v + "_cos"] = data[v].apply(lambda x: round(cos(2 * pi * x / x_max), 3))
    return


data['ARREST_MONTH'] = data['ARREST_DATE'].dt.month

encoding: dict[str, dict[str, int]] = {
    "LAW_CAT_CD": felony_misdemeanor,
    "PERP_SEX": male_female,
    "ARREST_BORO": arrest_boro,
    "AGE_GROUP": age_group,
    "ARREST_MONTH": monthly_val
}


df: DataFrame = data.replace(encoding, inplace=False)
encode_cyclic_variables(data, ["ARREST_MONTH"])

"""
Create date variables and drop the original date
"""
df['ARREST_DAY'] = df['ARREST_DATE'].dt.day
df['ARREST_YEAR'] = df['ARREST_DATE'].dt.year

df = df.drop(columns=['ARREST_DATE'])

"""
Delete columns as directly correlated with other variables
"""
df = df.drop(columns=['PD_DESC', 'OFNS_DESC'])

"""
Turn the LAW_CAT_CD into a discrete variable
"""
df['LAW_CAT_CD'] = df['LAW_CAT_CD'].astype('category')

"""
Label encoding
"""
# Assuming df is your DataFrame and 'LAW_CODE' is the column to encode
label_encoder = LabelEncoder()
df['LAW_CODE'] = label_encoder.fit_transform(df['LAW_CODE'])

#print(df['LAW_CODE'])

df.to_csv(f"./Data/{file_tag}_encoding.csv", index=True)