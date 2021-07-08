# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.base import clone
from julearn.utils.array import ensure_2d
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler

from julearn.transformers import (
    ConfoundRemover, get_transformer
)
from julearn.pipeline import make_pipeline

X = pd.DataFrame(dict(A=np.arange(10),
                      B=np.arange(10, 20),
                      C=np.arange(30, 40)
                      ))
y = pd.Series(np.arange(50, 60))

X_with_types = pd.DataFrame({
    'a__:type:__continuous': np.arange(10),
    'b__:type:__continuous': np.arange(10, 20),
    'c__:type:__confound': np.arange(30, 40),
    'd__:type:__confound': np.arange(40, 50),
    'e__:type:__categorical': np.arange(40, 50),
    'f__:type:__categorical': np.arange(40, 50),
})


def test_create_pipeline_steps_added_correctly():
    """test pipeline creation"""
    # test whether the steps are added
    # and whether all the hyperparameters were transferred correctly
    scaler = StandardScaler(with_mean=False)
    pca = PCA(n_components=3)
    lr = LinearRegression()
    steps = [('zscore', scaler),
             ('pca', pca),
             ('linear_reg', lr)]

    my_pipeline = make_pipeline(steps)

    for my_step, original_estimator in zip(
            my_pipeline.steps, [scaler, pca, lr]):
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_step[1].get_params().get(est_param) == est_val

    y_transformer = StandardScaler()
    my_pipeline = make_pipeline(steps, y_transformer=y_transformer)

    for my_step, original_estimator in zip(
            my_pipeline.steps, [y_transformer, scaler, pca, lr]):
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_step[1].get_params().get(est_param) == est_val

    pca_conf = PCA(10)
    confound_steps = [('pca', pca_conf)]
    my_pipeline = make_pipeline(
        steps, confound_steps=confound_steps, y_transformer=y_transformer)

    for my_step, original_estimator in zip(
            my_pipeline.steps, [pca_conf, y_transformer, scaler, pca, lr]):
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_step[1].get_params().get(est_param) == est_val


def test_fit_and_score():
    """Test fitting and scoring"""
    steps = [('zscore', StandardScaler()),
             ('pca', PCA()),
             ('lr', LinearRegression())]

    confound_steps = [('zscore', StandardScaler())]
    y_transformer = StandardScaler()
    extended_pipe = make_pipeline(
        steps, confound_steps=confound_steps,
        y_transformer=y_transformer)

    extended_pipe.fit(X.iloc[:, :-1], X.C, confounds=X.B)
    extended_pipe.predict(X.iloc[:, :-1])
    score = extended_pipe.score(X.iloc[:, :-1], X.C)
    assert score is not np.nan


def test_access_steps_ExtendedPipeline():
    """test access named_steps and named_confound_steps"""
    steps = [('zscore', StandardScaler()),
             ('pca', PCA()),
             ('lr', LinearRegression())
             ]
    y_transformer = StandardScaler()
    my_pipe = make_pipeline(
        steps,
        y_transformer=y_transformer,
        confound_steps=steps[0:1])

    assert (my_pipe.named_confound_steps.zscore  # type: ignore
            == my_pipe['confound__zscore']
            == my_pipe.confound_pipeline.named_steps.zscore  # type: ignore
            )

    assert (my_pipe.named_steps.pca
            == my_pipe['pca']
            == my_pipe.pipeline.named_steps.pca
            )

    assert (my_pipe.named_steps.lr
            == my_pipe['lr']
            == (my_pipe.pipeline.named_steps.lr)
            )

    with pytest.raises(ValueError, match='Indexing must be done '):
        my_pipe[0]


def test_preprocess_all_ExtendedPipeline():
    feature_steps = [
        ('zscore', StandardScaler()),
        ('pca', PCA())
    ]
    # model = ('lr', LinearRegression())

    confound_steps = [('zscore', StandardScaler()),
                      ('zscore_2', StandardScaler())]

    y_transformer = StandardScaler()

    feature_pipe = Pipeline(feature_steps)
    confounds_pipe = Pipeline(confound_steps)
    extended_pipe = make_pipeline(
        feature_steps,
        y_transformer=y_transformer,
        confound_steps=confound_steps)

    np.random.seed(42)
    X_trans_preprocess = extended_pipe.fit_transform(X, y, confounds=X.B)

    np.random.seed(42)
    conf_trans = confounds_pipe.fit_transform(ensure_2d(X.B), y)
    y_trans = y_transformer.fit_transform(ensure_2d(y)).squeeze()
    X_trans = feature_pipe.fit_transform(X, y_trans)

    _, y_trans_preprocess, conf_preprocess = \
        extended_pipe.preprocess(X, y, until='pca', confounds=X.B)

    assert_array_equal(X_trans, X_trans_preprocess)
    assert_array_equal(conf_trans, conf_preprocess)
    assert_array_equal(y_trans, y_trans_preprocess)


def test_preprocess_until_ExtendedDataFramePipeline():
    feature_steps = [('zscore', StandardScaler()),
                     ('other', RobustScaler()),
                     ]

    confound_steps = [('zscore', StandardScaler()),
                      ('zscore_2', StandardScaler())]

    y_transformer = StandardScaler()
    extended_pipe = make_pipeline(
        feature_steps,
        y_transformer=clone(y_transformer),
        confound_steps=confound_steps)

    np.random.seed(42)
    extended_pipe.fit(X, y, confounds=X.B)
    np.random.seed(42)
    X_trans = X.copy()
    y_trans = y.copy()
    conf_trans = ensure_2d(X.B)

    for name, step in confound_steps:

        this_confounds_pipe = make_pipeline(steps=[(name, step)])
        conf_trans = this_confounds_pipe.fit_transform(conf_trans)

        X_trans_pipe, y_trans_pipe, conf_trans_pipe = extended_pipe.preprocess(
            X, y, until=f'confound__{name}', confounds=X.B)

        assert_array_equal(X, X_trans_pipe)
        assert_array_equal(y, y_trans_pipe)
        assert_array_equal(conf_trans, conf_trans_pipe)

    y_trans = y_transformer.fit_transform(ensure_2d(y_trans)).squeeze()
    X_trans_pipe, y_trans_pipe, conf_trans_pipe = extended_pipe.preprocess(
        X, y, until='target__', confounds=X.B)
    assert_array_equal(X, X_trans_pipe)
    assert_array_equal(y_trans, y_trans_pipe)
    assert_array_equal(conf_trans, conf_trans_pipe)

    for name, step in feature_steps:
        this_feature_pipe = make_pipeline(steps=[(name, step)])
        X_trans = this_feature_pipe.fit_transform(X_trans)
        X_trans_pipe, y_trans_pipe, conf_trans_pipe = \
            extended_pipe.preprocess(X, y, until=name, confounds=X.B)

        assert_array_equal(X_trans, X_trans_pipe)
        assert_array_equal(y_trans, y_trans_pipe)
        assert_array_equal(conf_trans, conf_trans_pipe)

    with pytest.raises(ValueError, match='banana_pie is not a valid'):
        extended_pipe.preprocess(X, y, until='banana_pie', confounds=X.B)


def test_create_exteneded_pipeline_confound_removal():
    """Test pipeline with confound remover"""
    preprocess_steps_feature = [
        ('remove_confound', ConfoundRemover()),
        ('lr', LinearRegression())
    ]

    conf_steps = [('zscore', StandardScaler())]
    y_transformer = StandardScaler()

    extended_pipe = make_pipeline(
        steps=preprocess_steps_feature,
        y_transformer=y_transformer,
        confound_steps=conf_steps
    )

    np.random.seed(4242)
    extended_pipe.fit(
        X.iloc[:, :-1], X.C, confounds=X.B).predict(
            X.iloc[:, :-1], confounds=X.B)


def test_tune_params():
    params = {
        'svm__kernel': 'linear',
        'zscore__with_mean': True,
        'confounds__zscore__with_mean': True,
        'target__with_mean': True}

    extended_pipe = make_pipeline(
        steps=[
            ('zscore', get_transformer('zscore')),
            ('svm', SVR())
        ],
        confound_steps=[('zscore', get_transformer('zscore'))],
        y_transformer=get_transformer('zscore')
    )

    extended_pipe.set_params(**params)
    for param, val in params.items():
        assert extended_pipe.get_params()[param] == val

    with pytest.raises(ValueError, match='Each element of the'):
        extended_pipe.set_params(cOnFouunds__zscore__with_mean=True)


def test_ExtendedPipeline___repr__():
    extended_pipe = make_pipeline(
        steps=[
            ('zscore', get_transformer('zscore')),
            ('svm', SVR())
        ],
        confound_steps=[('zscore', get_transformer('zscore'))],
        y_transformer=get_transformer('zscore')
    )
    extended_pipe.__repr__()
