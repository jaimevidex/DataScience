from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types
from sklearn.preprocessing import LabelEncoder

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

encoding: dict[str, dict[str, int]] = {
    "LAW_CAT_CD": felony_misdemeanor,
    "PERP_SEX": male_female,
    "PERP_RACE": perp_race,
    "ARREST_BORO": arrest_boro,
    "AGE_GROUP": age_group
}

df: DataFrame = data.replace(encoding, inplace=False)

"""
Create date variables and drop the original date
"""
df['ARREST_DAY'] = df['ARREST_DATE'].dt.day
df['ARREST_MONTH'] = df['ARREST_DATE'].dt.month
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

print(df['LAW_CODE'])

df.to_csv(f"./Data/{file_tag}_encoding.csv", index=True)