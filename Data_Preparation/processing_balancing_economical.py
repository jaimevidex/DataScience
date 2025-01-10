from pandas import read_csv, concat, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_bar_chart
from numpy import ndarray
from pandas import Series
from imblearn.over_sampling import SMOTE

filename = "./Data/train_class_financial_distress_scaled_zscore.csv"
file = "class_financial_distress"
file_tag = "class_financial_distress_class_balance"
target = "CLASS"
original: DataFrame = read_csv(filename, sep=",", decimal=".")

target_count: Series = original[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

print("Minority class=", positive_class, ":", target_count[positive_class])
print("Majority class=", negative_class, ":", target_count[negative_class])
print(
    "Proportion:",
    round(target_count[positive_class] / target_count[negative_class], 2),
    ": 1",
)
values: dict[str, list] = {
    "Original": [target_count[positive_class], target_count[negative_class]]
}
figure()
plot_bar_chart(
    target_count.index.to_list(), target_count.to_list(), title="Class balance"
)
savefig(f"Data_Preparation/Images/{file_tag}_eval.png")
show()

""" Oversampling """

df_positives: Series = original[original[target] == positive_class]
df_negatives: Series = original[original[target] == negative_class]

df_pos_sample: DataFrame = DataFrame(
    df_positives.sample(len(df_negatives), replace=True)
)
df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
df_over.to_csv(f"data/{file}_over.csv", index=False)

print("Minority class=", positive_class, ":", len(df_pos_sample))
print("Majority class=", negative_class, ":", len(df_negatives))
print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")


""" SMOTE """

RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = original.pop(target).values
X: ndarray = original.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(original.columns) + [target]
df_smote.to_csv(f"data/{file}_smote.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)

""" Undersampling """

df_neg_sample: DataFrame = DataFrame(
    df_negatives.sample(len(df_positives), replace=False)
)
df_under: DataFrame = concat([df_neg_sample, df_positives], axis=0)
df_under.to_csv(f"data/{file}_under.csv", index=False)

print("Minority class=", positive_class, ":", len(df_positives))
print("Majority class (after undersampling)=", negative_class, ":", len(df_neg_sample))
print("Proportion (undersampled):", round(len(df_positives) / len(df_neg_sample), 2), ": 1")