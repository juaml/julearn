# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    SelectPercentile,
    SelectKBest,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    VarianceThreshold,
)
from sklearn.base import BaseEstimator, TransformerMixin
from seaborn import load_dataset

import pytest

from julearn.utils.testing import (
    do_scoring_test,
    PassThroughTransformer,
)
from julearn.transformers import (
    get_transformer,
    reset_transformer_register,
    register_transformer,
    DataFrameConfoundRemover,
)

from julearn.transformers.available_transformers import (
    _get_returned_features,
    _get_apply_to,
)


class fish(BaseEstimator, TransformerMixin):
    def __init__(self, can_it_fly):
        self.can_it_fly = can_it_fly

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


reset_transformer_register()


@pytest.mark.parametrize(
    "name, klass, params",
    [
        ("zscore", StandardScaler, {}),
        ("scaler_robust", RobustScaler, {}),
        ("scaler_minmax", MinMaxScaler, {}),
        ("scaler_maxabs", MaxAbsScaler, {}),
        ("scaler_normalizer", Normalizer, {}),
        ("scaler_quantile", QuantileTransformer, {"n_quantiles": 10}),
        ("scaler_power", PowerTransformer, {}),
        ("select_univariate", GenericUnivariateSelect, {}),
        ("select_percentile", SelectPercentile, {}),
        ("select_k", SelectKBest, {"k": 2}),
        ("select_fdr", SelectFdr, {}),
        ("select_fpr", SelectFpr, {}),
        ("select_fwe", SelectFwe, {}),
        ("select_variance", VarianceThreshold, {}),
    ],
)
def test_feature_transformers(name, klass, params):
    """Test transform X"""
    """Test simple binary classification"""
    df_iris = load_dataset("iris")

    # keep only two species
    df_iris = df_iris[df_iris["species"].isin(["setosa", "virginica"])]
    X = ["sepal_length", "sepal_width", "petal_length"]
    y = "species"

    scorers = ["accuracy"]

    api_params = {"model": "svm", "preprocess": name}
    model_params = {f"{name}__{k}": v for k, v in params.items()}
    api_params["model_params"] = model_params
    tr = klass(**params)
    clf = make_pipeline(tr, svm.SVC())
    df_test = df_iris.copy()
    do_scoring_test(
        X,
        y,
        data=df_test,
        api_params=api_params,
        sklearn_model=clf,
        scorers=scorers,
    )


def test_wrong_target_transformers():
    """Test wrong target transformers"""
    with pytest.raises(ValueError, match="is not available"):
        get_transformer("scaler_robust", target=True)

    with pytest.raises(ValueError, match="is not available"):
        get_transformer("wrong", target=False)


def test__get_apply_to():
    apply_to_confound = _get_apply_to(DataFrameConfoundRemover())
    apply_to_select = _get_apply_to(get_transformer("select_percentile"))
    apply_to_zscore = _get_apply_to(StandardScaler())

    with pytest.warns(
        RuntimeWarning,
        match=("is not a registered " "transformer. " "Therefore, `apply_to`"),
    ):
        apply_to_pass = _get_apply_to(PassThroughTransformer())
    assert apply_to_zscore == apply_to_pass == "continuous"
    assert apply_to_confound == ["continuous", "confound"]
    assert apply_to_select == "all_features"


def test_register_reset():
    reset_transformer_register()
    with pytest.raises(ValueError, match="The specified transformer"):
        get_transformer("passthrough")

    register_transformer("passthrough", PassThroughTransformer, "same", "all")
    assert get_transformer("passthrough").__class__ == PassThroughTransformer
    assert _get_apply_to(PassThroughTransformer()) == "all"
    assert _get_returned_features(PassThroughTransformer()) == "same"

    with pytest.warns(RuntimeWarning, match="Transformer named"):
        register_transformer(
            "passthrough", PassThroughTransformer, "same", "all"
        )
    reset_transformer_register()
    with pytest.raises(ValueError, match="The specified transformer"):
        get_transformer("passthrough")

    register_transformer(
        "passthrough", PassThroughTransformer, "unknown", "continuous"
    )
    assert get_transformer("passthrough").__class__ == PassThroughTransformer
    assert _get_apply_to(PassThroughTransformer()) == "continuous"
    assert _get_returned_features(PassThroughTransformer()) == "unknown"


def test_register_class_no_default_params():

    reset_transformer_register()
    register_transformer("fish", fish, "unknown", "all")
    get_transformer("fish", can_it_fly="dont_be_stupid")


def test_get_target_transformer_no_error():
    get_transformer("zscore", target=True)
    get_transformer("remove_confound", target=True)


def test_register_warning():
    with pytest.warns(RuntimeWarning, match="Transformer name"):
        register_transformer("zscore", fish, "unknown", "all")
    reset_transformer_register()

    with pytest.raises(ValueError, match="Transformer name"):
        register_transformer("zscore", fish, "unknown", "all", overwrite=False)
    reset_transformer_register()

    with pytest.warns(None) as record:
        register_transformer("zscore", fish, "unknown", "all", overwrite=True)

    reset_transformer_register()
    assert len(record) == 0
