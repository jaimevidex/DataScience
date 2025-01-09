from pandas import Series,  read_csv, DataFrame
from sklearn.base import RegressorMixin
from matplotlib.pyplot import savefig
from dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series

class PersistenceOptimistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last: float = 0.0
        return

    def fit(self, X: Series):
        self.last = X.iloc[-1]
        # print(self.last)
        return

    def predict(self, X: Series):
        prd: list = X.shift().values.ravel()
        prd[0] = self.last
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series
    
train_filename: str = "Data/train_forecast_ny_arrests_differentiated.csv"
test_filename: str = "Data/test_forecast_ny_arrests_differentiated.csv"
filename: str = "Data/forecast_ny_arrests.csv"
file_tag: str = "NYC"
target: str = "Manhattan"
timecol: str = "Date"

train_data: DataFrame = read_csv(train_filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)
test_data: DataFrame = read_csv(test_filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)

train: Series = train_data[target]
test: Series = test_data[target]

fr_mod = PersistenceOptimistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Optimist")
savefig(f"Forecasting/Images/Persistence/{file_tag}_persistence_optim_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Persistence Optimist",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"Forecasting/Images/Persistence/{file_tag}_persistence_optim_forecast.png")

class PersistenceRealistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0
        self.estimations = [0]
        self.obs_len = 0

    def fit(self, X: Series):
        for i in range(1, len(X)):
            self.estimations.append(X.iloc[i - 1])
        self.obs_len = len(self.estimations)
        self.last = X.iloc[len(X) - 1]
        prd_series: Series = Series(self.estimations)
        prd_series.index = X.index
        return prd_series

    def predict(self, X: Series):
        prd: list = len(X) * [self.last]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series
fr_mod = PersistenceRealistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Realist")
savefig(f"Forecasting/Images/Persistence/{file_tag}_persistence_real_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Persistence Realist",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"Forecasting/Images/Persistence/{file_tag}_persistence_real_forecast.png")