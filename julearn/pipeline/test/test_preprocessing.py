from julearn.pipeline.preprocessing import (
    make_type_selector,
    PreprocessCreator,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from julearn.transformers import get_transformer


def test_make_type_select_continuous(X_typed_iris, df_typed_iris):
    selector = make_type_selector("__:type:__continuous")
    X_selected = selector(X_typed_iris)
    df = df_typed_iris.rename(
        columns=dict(species="s__:type:__chicken")
    )
    df_selected = selector(df)
    assert X_selected == df_selected


def test_make_type_select_chicken(X_typed_iris, df_typed_iris):
    selector = make_type_selector("__:type:__chicken")
    X_selected = selector(X_typed_iris)
    df = df_typed_iris.rename(
        columns=dict(species="s__:type:__chicken")
    )
    df_selected = selector(df)

    assert len(X_selected) == 0
    assert df_selected == ["s__:type:__chicken"]


def test_preprocessor_add(preprocessing):
    preprocessor = PreprocessCreator()
    for transformer in preprocessing:
        preprocessor.add(transformer)

    for preprocess, step in zip(preprocessing, preprocessor.steps):
        name, transformer = step
        assert name.startswith(f"wrapped_{preprocess}")
        assert isinstance(transformer, ColumnTransformer)
        assert isinstance(
            transformer.transformers[0][1],
            get_transformer(preprocess).__class__)


def test_preprocessor_from_list(preprocessing):
    preprocessor = PreprocessCreator.from_list(
        preprocessing
    )

    for preprocess, step in zip(preprocessing, preprocessor.steps):
        name, transformer = step
        assert name.startswith(f"wrapped_{preprocess}")
        assert isinstance(transformer, ColumnTransformer)
        assert isinstance(
            transformer.transformers[0][1],
            get_transformer(preprocess).__class__)


def test_preprocessor_add_hyperparams(
        X_typed_iris, y_typed_iris,
        preprocessing, get_default_params):
    preprocessor = PreprocessCreator()
    for transformer in preprocessing:
        preprocessor.add(
            transformer,
            **get_default_params(transformer)
        )
    pipe = Pipeline(
        [*preprocessor.steps, ("rf", RandomForestClassifier())]
    ).set_output(transform="pandas")
    model_params = preprocessor.param_grid
    pipe = GridSearchCV(pipe, param_grid=model_params)
    pipe.fit(X_typed_iris, y_typed_iris)


def test_wrap_step():
    column_transformer = (
        PreprocessCreator()
        .wrap_step("cheese", "zscore", "__:type:__continuous")
    )
    name, trans_name, _ = column_transformer.transformers[0]
    assert name == "cheese"
    assert trans_name == "zscore"


def test_names_same_transformer():
    steps = PreprocessCreator.from_list(["zscore", "pca", "zscore"]).steps
    names = [name for name, _ in steps]
    assert names == ["wrapped_zscore", "wrapped_pca", "wrapped_zscore_1"]
