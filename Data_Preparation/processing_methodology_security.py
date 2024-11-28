from numpy import ndarray
from pandas import DataFrame, read_csv
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN
from sklearn.naive_bayes import _BaseNB, GaussianNB, BernoulliNB
from imblearn.over_sampling import SMOTE 
from joblib import Parallel, delayed

DELTA_IMPROVE: float = 0.001

#def run_NB1(trnX, trnY, tstX, tstY, metric: str = "accuracy") -> dict[str, float]:
#    estimators: dict[str, GaussianNB | BernoulliNB] = {
#        "GaussianNB": GaussianNB(),
#        "BernoulliNB": BernoulliNB(),
#    }
#    best_model: GaussianNB | BernoulliNB = None  # type: ignore
#    best_performance: float = 0.0
#    eval: dict[str, float] = {}
#
#
#    for clf in estimators:
#        estimators[clf].fit(trnX, trnY)
#        prdY: ndarray = estimators[clf].predict(tstX)
#        performance: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
#        if performance - best_performance > DELTA_IMPROVE:
#            best_performance = performance
#            best_model = estimators[clf]
#    if best_model is not None:
#        prd: ndarray = best_model.predict(tstX)
#        for key in CLASS_EVAL_METRICS:
#            eval[key] = CLASS_EVAL_METRICS[key](tstY, prd)
#    return eval
#
#def evaluate_approach(
#    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
#) -> dict[str, list]:
#    trnY = train.pop(target).values
#    trnX: ndarray = train.values
#    tstY = test.pop(target).values
#    tstX: ndarray = test.values
#    eval: dict[str, list] = {}
#
#    eval_NB: dict[str, float] = run_NB1(trnX, trnY, tstX, tstY, metric=metric)
#    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
#    if eval_NB != {} and eval_KNN != {}:
#        for met in CLASS_EVAL_METRICS:
#            eval[met] = [eval_NB[met], eval_KNN[met]]
#    return eval

def run_NB1(trnX, trnY, tstX, tstY, metric: str = "accuracy") -> dict[str, float]:
    estimator = GaussianNB()  # Choose the most appropriate NB variant for your data
    
    estimator.fit(trnX, trnY)
    prdY: ndarray = estimator.predict(tstX)
    
    eval: dict[str, float] = {}
    for key in CLASS_EVAL_METRICS:
        eval[key] = CLASS_EVAL_METRICS[key](tstY, prdY)
    
    return eval

def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    
    results = Parallel(n_jobs=-1)(
        delayed(func)(trnX, trnY, tstX, tstY, metric=metric)
        for func in [run_NB1, run_KNN]
    )
    
    eval_NB, eval_KNN = results
    
    eval: dict[str, list] = {
        met: [eval_NB[met], eval_KNN[met]] for met in CLASS_EVAL_METRICS
    }
    
    return eval

filename = "Data/class_ny_arrests.csv"
file_tag = "class_ny_arrests"
train: DataFrame = read_csv("Data/train_data_class_ny_arrests_zscore.csv")
test: DataFrame = read_csv("Data/test_data_class_ny_arrests_zscore.csv")

figure()
eval: dict[str, list] = evaluate_approach(train, test, target="LAW_CAT_CD", metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"./{file_tag}_eval_zscore.png")
show()