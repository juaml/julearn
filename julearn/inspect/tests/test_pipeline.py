import pytest
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from julearn.inspect import _EstimatorInspector, PipelineInspector
from julearn.pipeline import PipelineCreator
from julearn.transformers import JuColumnTransformer


class TestEst(BaseEstimator):
    def __init__(self, hype_0=0, hype_1=1):
        self.hype_0 = hype_0
        self.hype_1 = hype_1

    def fit(self, X, y=None, **fit_params):
        self.param_0_ = 0
        self.param_1_ = 1
        return self

    def transform(self, X):
        return X


@pytest.mark.parametrize(
    "steps", [
        ["svm"],
        ["zscore", "svm"],
        ["pca", "svm"],
        ["zscore", "pca", "svm"],
    ])
def test_get_stepnames(steps, df_iris):
    pipe = (PipelineCreator(problem_type="classification")
            .from_list(steps, model_params={}, problem_type="classification")
            .to_pipeline()
            )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    assert (["set_column_types"] + steps
            == PipelineInspector(pipe).get_step_names()
            )


@pytest.mark.parametrize(
    "steps,as_estimator,returns", [
        (["svm"], True, [SVC()]),
        (["zscore", "pca", "svm"], True, [StandardScaler(), PCA(), SVC()]),
        (["svm"], False, [_EstimatorInspector(SVC())]),

    ])
def test_steps(steps, as_estimator, returns, df_iris):

    pipe = (PipelineCreator(problem_type="classification")
            .from_list(steps, model_params={}, problem_type="classification")
            .to_pipeline()
            )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = PipelineInspector(pipe)
    for i, _ in enumerate(steps):
        step_est = inspector.get_step(name=steps[i], as_estimator=as_estimator)
        assert isinstance(step_est, returns[i].__class__)


@pytest.mark.parametrize(
    "est,fitted_params", [
        [TestEst(), {"param_0_": 0, "param_1_": 1}],
        [JuColumnTransformer("test", TestEst(), "continuous"),
         {"param_0_": 0, "param_1_": 1}],
    ])
def test_inspect_estimator(est, fitted_params, df_iris):

    est.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = _EstimatorInspector(est)
    assert est.get_params() == inspector.get_params()
    inspect_params = inspector.get_fitted_params()
    inspect_params.pop("column_transformer_", None)
    assert fitted_params == inspect_params


def test_inspect_pipeline(df_iris):

    expected_fitted_params = {
        "jucolumntransformer__param_0_": 0, "jucolumntransformer__param_1_": 1}

    pipe = (PipelineCreator(problem_type="classification")
            .add(JuColumnTransformer("test", TestEst(), "continuous"))
            .add(SVC())
            .to_pipeline()
            )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = PipelineInspector(pipe)
    inspect_params = inspector.get_fitted_params()
    inspect_params.pop("jucolumntransformer__column_transformer_", None)
    inspect_params = {
        key: val for key, val in inspect_params.items()
        if (not key.startswith("svc")) and (
            not key.startswith("set_column_types"))
    }

    assert expected_fitted_params == inspect_params


def test_get_estimator(df_iris):
    pipe = (PipelineCreator(problem_type="classification")
            .add(JuColumnTransformer("test", TestEst(), "continuous"))
            .add(SVC())
            .to_pipeline()
            )
    pipe.fit(df_iris.iloc[:, :-1], df_iris.species)
    inspector = PipelineInspector(pipe)
    svc = inspector.get_step("svc").estimator
    assert isinstance(svc, SVC)
    assert pipe.get_params() ==  inspector.get_params()
