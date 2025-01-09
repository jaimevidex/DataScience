from numpy import arange
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig
from sklearn.linear_model import LinearRegression
from dslabs_functions import series_train_test_split, plot_forecasting_eval

train_filename: str = "Data/train_forecast_ny_arrests_differentiated.csv"
test_filename: str = "Data/test_forecast_ny_arrests_differentiated.csv"
file_tag: str = "NYC"
target: str = "Manhattan"
timecol: str = "Date"

train_data: DataFrame = read_csv(train_filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)
test_data: DataFrame = read_csv(test_filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)
train: Series = train_data[target]
test: Series = test_data[target]

trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(train) + len(test)).reshape(-1, 1)
tstY = test.to_numpy()

model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX), index=train.index)
prd_tst: Series = Series(model.predict(tstX), index=test.index)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression")
savefig(f"Forecasting/Images/LinearRegression/{file_tag}_linear_regression_eval.png")

from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Linear Regression",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"Forecasting/Images/LinearRegression/{file_tag}_linear_regression_forecast.png")