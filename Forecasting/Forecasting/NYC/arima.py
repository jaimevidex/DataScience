from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA
from dslabs_functions import series_train_test_split, HEIGHT
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

train: Series = train_data
test: Series = test_data

target_train = train[target]
target_test = test[target]

predictor = ARIMA(target_train, exog=train, order=(3, 0, 7))
model = predictor.fit()
print(model.summary())
model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
savefig(f"Forecasting/Images/ARIMA/{file_tag}_multi_arima_{measure}_diagnostics.png")


from matplotlib.pyplot import figure, savefig, subplots
from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart

def arima_study_multi(train: Series, test: Series, measure: str = "R2"):
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                arima = ARIMA(target_train, exog=train , order=(p, d, q))
                model = arima.fit()
                prd_tst = model.forecast(exog=test, steps=len(test), signal_only=False)
                eval: float = FORECAST_MEASURES[measure](test[target], prd_tst)
                print(f"ARIMA ({p}, {d}, {q})", eval)
                if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params["params"] = (p, d, q)
                    best_model = model
                yvalues.append(eval)
            values[q] = yvalues
        plot_multiline_chart(
            p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
        )
    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

    return best_model, best_params

from matplotlib.pyplot import savefig

best_model, best_params = arima_study_multi(train, test, measure=measure)
savefig(f"Forecasting/Images/ARIMA/{file_tag}_multi_arima_{measure}_study.png")


from dslabs_functions import plot_forecasting_eval

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test), exog=test)

plot_forecasting_eval(
    target_train, target_test, prd_trn, prd_tst, title=f"{file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
)
savefig(f"Forecasting/Images/ARIMA/{file_tag}_multi_arima_{measure}_eval.png")

from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    target_train,
    target_test,
    prd_tst,
    title=f"{file_tag} - ARIMA ",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"Forecasting/Images/ARIMA/{file_tag}_multi_arima_{measure}_forecast.png")

predictor = ARIMA(target_train, order=(3, 1, 7))
model = predictor.fit()
print(model.summary())
model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
savefig(f"Forecasting/Images/ARIMA/{file_tag}_uni_arima_{measure}_diagnostics.png")


from matplotlib.pyplot import figure, savefig, subplots
from dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart

def arima_study_uni(train: Series, test: Series, measure: str = "R2"):
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                arima = ARIMA(train,order=(p, d, q))
                model = arima.fit()
                prd_tst = model.forecast(steps=len(test), signal_only=False)
                eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                print(f"ARIMA ({p}, {d}, {q})", eval)
                if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params["params"] = (p, d, q)
                    best_model = model
                yvalues.append(eval)
            values[q] = yvalues
        plot_multiline_chart(
            p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
        )
    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

    return best_model, best_params

from matplotlib.pyplot import savefig

best_model, best_params = arima_study_uni(target_train, target_test, measure=measure)
savefig(f"Forecasting/Images/ARIMA/{file_tag}_uni_arima_{measure}_study.png")


from dslabs_functions import plot_forecasting_eval

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(target_train) - 1)
prd_tst = best_model.forecast(steps=len(target_test))

plot_forecasting_eval(
    target_train, target_test, prd_trn, prd_tst, title=f"{file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
)
savefig(f"Forecasting/Images/ARIMA/{file_tag}_uni_arima_{measure}_eval.png")

from dslabs_functions import plot_forecasting_series

plot_forecasting_series(
    target_train,
    target_test,
    prd_tst,
    title=f"{file_tag} - ARIMA ",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"Forecasting/Images/ARIMA/{file_tag}_uni_arima_{measure}_forecast.png")