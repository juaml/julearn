from seaborn import load_dataset
from pytest import fixture
from copy import copy


@fixture(scope="module")
def df_typed_iris():
    df = load_dataset("iris")
    return df


@fixture(scope="module")
def X_iris():
    df = load_dataset("iris")
    return df.iloc[:, :-1]


@fixture(scope="module")
def y_iris():
    df = load_dataset("iris")
    return (
        df["species"]
        .map(lambda x: dict(setosa=0, versicolor=1, virginica=2)[x])
    )


@fixture(params=[None, dict(), dict(duck=["petal_length"]),
                 dict(duck=["petal_length"], confound=["petal_width"])],
         scope="module")
def X_types_iris(request):
    return request.param


@fixture(params=["rf", "svm", "gauss", "ridge"], scope="module")
def models_all_problem_types(request):
    return request.param


@fixture(params=["regression",
                 "classification"],
         scope="module"
         )
def all_problem_types(request):
    return request.param


step_to_params = dict(
    zscore=dict(with_mean=[True, False]),
    pca=dict(n_components=[.2, .7]),
    select_univariate=dict(mode=["k_best", "percentile"]),
    rf=dict(n_estimators=[2, 5]),
    svm=dict(C=[1, 2]),
    ridge=dict(alpha=[1, 2])
)


@fixture(scope="module")
def get_default_params():
    def get(step):
        return copy(step_to_params.get(step, {}))
    return get


@fixture(params=[
    "zscore", ["zscore"],
    ["pca"],
    ["select_univariate"],
    ["zscore", "pca"],
    ["select_univariate", "zscore", "pca"],
], scope="module")
def preprocessing(request):
    return request.param
