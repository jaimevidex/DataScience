from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from dslabs_functions import series_train_test_split, HEIGHT, FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart, plot_forecasting_series,plot_forecasting_eval
from matplotlib.pyplot import savefig

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



def exponential_smoothing_study(train: Series, test: Series, measure: str = "R2"):
    alpha_values = [i / 10 for i in range(1, 10)]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for alpha in alpha_values:
        tool = SimpleExpSmoothing(train)
        model = tool.fit(smoothing_level=alpha, optimized=False)
        prd_tst = model.forecast(steps=len(test))

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (alpha,)
            best_model = model
        yvalues.append(eval)

    print(f"Exponential Smoothing best with alpha={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing ({measure})",
        xlabel="alpha",
        ylabel=measure,
        percentage=flag,
    )

    return best_model, best_params

best_model, best_params = exponential_smoothing_study(train, test, measure=measure)
savefig(f"Forecasting/Images/ExponentialSmoothing/{file_tag}_exponential_smoothing_{measure}_study.png")

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Exponential Smoothing alpha={params[0]}")
savefig(f"Forecasting/Images/ExponentialSmoothing/{file_tag}_exponential_smoothing_{measure}_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Exponential Smoothing ",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"Forecasting/Images/ExponentialSmoothing/{file_tag}_exponential_smoothing_{measure}_forecast.png")