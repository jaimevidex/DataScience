from numpy import mean
from pandas import Series
from sklearn.base import RegressorMixin
from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig
from dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series, HEIGHT

class RollingMeanRegressor(RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win
        self.memory: list = []

    def fit(self, X: Series):
        self.memory = X.iloc[-self.win_size :]
        # print(self.memory)
        return

    def predict(self, X: Series):
        estimations = self.memory.tolist()
        predictions = []

        for _ in range(len(X)):
            new_value = mean(estimations[-self.win_size:])
            estimations.append(new_value)
            predictions.append(new_value)

        # Create a Series with the correct index
        prd_series = Series(predictions, index=X.index)
        return prd_series
    

def rolling_mean_study(train: Series, test: Series, measure: str = "R2"):
    # win_size = (3, 5, 10, 15, 20, 25, 30, 40, 50)
    win_size = (12, 24, 48, 96, 192, 384, 768)
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Rolling Mean", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for w in win_size:
        pred = RollingMeanRegressor(win=w)
        pred.fit(train)
        prd_tst = pred.predict(test)

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (w,)
            best_model = pred
        yvalues.append(eval)

    print(f"Rolling Mean best with win={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        win_size, yvalues, title=f"Rolling Mean ({measure})", xlabel="window size", ylabel=measure, percentage=flag
    )

    return best_model, best_params

from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig
from dslabs_functions import series_train_test_split, plot_forecasting_eval, plot_forecasting_series, HEIGHT

train_filename: str = "Data/train_forecast_ny_arrests_differentiated.csv"
test_filename: str = "Data/test_forecast_ny_arrests_differentiated.csv"
filename: str = "Data/forecast_ny_arrests.csv"
file_tag: str = "NYC"
target: str = "Manhattan"
timecol: str = "Date"
measure: str = "R2"

train_data: DataFrame = read_csv(train_filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)
test_data: DataFrame = read_csv(test_filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)

train: Series = train_data[target]
test: Series = test_data[target]

fig = figure(figsize=(HEIGHT, HEIGHT))
best_model, best_params = rolling_mean_study(train, test)
savefig(f"Forecasting/Images/RollingMean/{file_tag}_rollingmean_{measure}_study.png")

params = best_params["params"]
prd_trn: Series = best_model.predict(train)
prd_tst: Series = best_model.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Rolling Mean (win={params[0]})")
savefig(f"Forecasting/Images/RollingMean/{file_tag}_rollingmean_{measure}_win{params[0]}_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Rolling Mean (win={params[0]})",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"Forecasting/Images/RollingMean/{file_tag}_rollingmean_{measure}_forecast.png")