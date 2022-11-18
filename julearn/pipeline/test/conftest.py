from seaborn import load_dataset
from pytest import fixture
from copy import copy


@fixture
def df_typed_iris():
    df = load_dataset("iris")
    X_names = df.iloc[:, :-1].columns.tolist()

    renamer = {col: col + "__:type:__continuous"
               for col in X_names}

    df = df.rename(columns=renamer)
    return df


@fixture
def X_typed_iris():
    df = load_dataset("iris")
    X_names = df.iloc[:, :-1].columns.tolist()

    renamer = {col: col + "__:type:__continuous"
               for col in X_names}

    df = df.rename(columns=renamer)
    return df[renamer.values()]


@fixture
def y_typed_iris():
    df = load_dataset("iris")
    return (
        df["species"]
        .map(lambda x: dict(setosa=0, versicolor=1, virginica=2)[x])
    )


@fixture(params=["rf", "svm", "gauss", "ridge"])
def models_all_problem_types(request):
    return request.param


@fixture(params=["regression",
                 "classification"])
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


@fixture
def get_default_params():
    def get(step):
        return copy(step_to_params[step])
    return get


@fixture(params=[
    ["zscore"], ["pca"],
    ["select_univariate"],
    ["zscore", "pca"],
    ["select_univariate", "zscore", "pca"],
])
def preprocessing(request):
    return request.param
