# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from julearn.transformers.confounds import TargetConfoundRemover
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

from julearn.utils.array import ensure_2d
from julearn.transformers.target import TargetTransformerWrapper
from julearn.transformers import (
    ConfoundRemover, get_transformer
)
from julearn.pipeline import make_pipeline

X = pd.DataFrame(
    dict(A=np.arange(10), B=np.arange(10, 20), C=np.arange(30, 40)))
y = pd.Series(np.arange(50, 60))
y_bin = pd.Series([0, 1] * 5)

X_with_types = pd.DataFrame({
    'a': np.arange(10),
    'b': np.arange(10, 20),
    'c': np.arange(30, 40),
    'd': np.arange(40, 50),
    'e': np.arange(40, 50),
    'f': np.arange(40, 50),
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
        my_trans = my_step[1]
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_trans.get_params().get(est_param) == est_val

    y_transformer = TargetTransformerWrapper(StandardScaler())
    my_pipeline = make_pipeline(steps, y_transformer=y_transformer)

    for my_step, original_estimator in zip(
            my_pipeline.steps, [y_transformer, scaler, pca, lr]):
        if my_step[0].startswith('target__'):
            continue
        my_trans = my_step[1]
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_trans.get_params().get(est_param) == est_val

    pca_conf = PCA(10)
    confound_steps = [('pca', pca_conf)]
    my_pipeline = make_pipeline(
        steps, confound_steps=confound_steps, y_transformer=y_transformer)

    for my_step, original_estimator in zip(
            my_pipeline.steps, [pca_conf, y_transformer, scaler, pca, lr]):
        if my_step[0].startswith('target__'):
            continue
        my_trans = my_step[1]
        if isinstance(my_trans, ColumnTransformer):
            my_trans = my_step[1].transformers[0][1]
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_trans.get_params().get(est_param) == est_val


def test_fit_and_score_no_error():
    steps = [('zscore', StandardScaler()),
             ('pca', PCA()),
             ('lr', LinearRegression())]

    y_transformer = StandardScaler()
    extended_pipe = make_pipeline(
        steps, y_transformer=y_transformer)

    extended_pipe.fit(X.iloc[:, :-1], X.C)
    extended_pipe.predict(X.iloc[:, :-1])
    score = extended_pipe.score(X.iloc[:, :-1], X.C)
    assert score is not np.nan


def test_prediction_binary():
    steps = [('log', LogisticRegression())]
    extended_pipe = make_pipeline(steps)
    lg = LogisticRegression()

    np.random.seed(24)
    extended_pipe.fit(X, y_bin)
    extended_proba = extended_pipe.predict_proba(X)
    extended_pred = extended_pipe.predict(X)
    extended_decision_func = extended_pipe.decision_function(X)
    extended_score = extended_pipe.score(X, y_bin)

    np.random.seed(24)
    lg.fit(X, y_bin)
    lg_proba = lg.predict_proba(X)
    lg_pred = lg.predict(X)
    lg_decision_func = lg.decision_function(X)
    lg_score = lg.score(X, y_bin)

    assert_array_equal(lg_proba, extended_proba)
    assert_array_equal(lg_pred, extended_pred)
    assert_array_equal(lg_decision_func, extended_decision_func)
    assert_array_equal(lg_score, extended_score)


def test_fit_and_score_confound_no_error():
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


def test_fit_then_transform():
    steps = [('zscore', StandardScaler()),
             ('pca', PCA()), ('lr', 'passthrough')
             ]

    y_transformer = StandardScaler()
    extended_pipe = make_pipeline(steps, y_transformer=y_transformer)
    np.random.seed(42)
    extended_trans = (extended_pipe
                      .fit(X.iloc[:, :-1], X.C)
                      .transform(X.iloc[:, :-1])
                      )

    np.random.seed(42)
    sk_pipe = Pipeline(steps)
    sk_trans = (sk_pipe
                .fit(X.iloc[:, :-1], X.C)
                .transform(X.iloc[:, :-1])
                )

    assert_array_equal(extended_trans, sk_trans)


def test_fit_transform():
    steps = [('zscore', StandardScaler()),
             ('pca', PCA()), ('lr', 'passthrough')
             ]

    y_transformer = StandardScaler()
    extended_pipe = make_pipeline(steps, y_transformer=y_transformer)
    np.random.seed(42)
    extended_trans = (extended_pipe
                      .fit_transform(X.iloc[:, :-1], X.C)
                      )

    np.random.seed(42)
    sk_pipe = Pipeline(steps)
    sk_trans = (sk_pipe
                .fit_transform(X.iloc[:, :-1], X.C)
                )

    assert_array_equal(extended_trans, sk_trans)


def test_fit_predict():
    steps = [('remove_confound', ConfoundRemover()),
             ('z-score', StandardScaler()),
             ('gmixed', GaussianMixture())
             ]
    extended_pipe = make_pipeline(steps)
    sk_pipe = Pipeline(steps[1:])
    np.random.seed(42)
    extended_pred = (extended_pipe
                     .fit_predict(X.iloc[:, :-1], X.C, n_confounds=1)
                     )
    np.random.seed(42)
    X_conf_rem = (ConfoundRemover()
                  .fit_transform(X.iloc[:, :-1], X.C, n_confounds=1)
                  )
    sk_pred = (sk_pipe
               .fit_predict(X_conf_rem, X.C)
               )

    assert_array_equal(extended_pred, sk_pred)


def test_multiple_confound_removal():

    my_pipe_drop = make_pipeline(
        steps=[
            ('remove_confound_1', ConfoundRemover()),
            ('remove_confound_2', ConfoundRemover()),
        ],
    )

    my_pipe_one_drop = make_pipeline(
        steps=[
            ('remove_confound_1', ConfoundRemover(drop_confounds=False)),
            ('remove_confound_2', ConfoundRemover(drop_confounds=True)),
        ],
    )

    my_pipe_no_drop = make_pipeline(
        steps=[
            ('remove_confound_1', ConfoundRemover(drop_confounds=False)),
            ('remove_confound_2', ConfoundRemover(drop_confounds=False)),
        ],
    )

    X_trans_one = (my_pipe_one_drop
                   .fit_transform(X, y, n_confounds=1)
                   )
    X_trans_no_1 = (clone(my_pipe_no_drop)
                    .fit(X, y, n_confounds=1).transform(X)
                    )
    X_trans_no_2 = (my_pipe_no_drop
                    .fit_transform(X, y, n_confounds=1)
                    )

    with pytest.warns(RuntimeWarning,
                      match='Number of confounds is 0'
                      ):
        my_pipe_drop.fit(X, y, n_confounds=1)

    assert_array_equal(X_trans_one.shape, X.iloc[:, :-1].shape)
    assert_array_equal(X_trans_no_1.shape, X.iloc[:, :-1].shape)
    assert_array_equal(X_trans_no_1.shape, X_trans_no_2.shape)


def test_target_removal_pipe():
    my_pipe = make_pipeline(
        steps=[('lr', LinearRegression())],
        y_transformer=TargetConfoundRemover()
    )
    my_pipe_no_rem = make_pipeline(
        steps=[('lr', LinearRegression())]
    )
    np.random.seed(42)
    y_trans_conf = TargetConfoundRemover().fit_transform(X, y, 1)

    np.random.seed(42)
    y_trans_conf_pipe = my_pipe.fit(X, y, n_confounds=1).transform_target(X, y)

    np.random.seed(42)
    y_trans_no_rem = my_pipe_no_rem.fit(X, y).transform_target(X, y)

    assert_array_equal(y, y_trans_no_rem)
    assert_array_equal(y_trans_conf_pipe, y_trans_conf)


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
    my_pipe.fit(X, y, n_confounds=1)
    assert (
        my_pipe.named_steps.confounds__zscore  # type: ignore
        == my_pipe['confounds__zscore']
        == my_pipe._confound_pipeline.named_steps.zscore  # type: ignore
    )
    assert (my_pipe.named_steps.pca
            == my_pipe['pca']
            # == (my_pipe._pipeline.named_steps
            #     ._internally_wrapped_pca.transformers[0][1])
            )
    assert (my_pipe.named_steps.pca.get_params() ==
            my_pipe._pipeline.named_steps
            ._internally_wrapped_pca.transformers[0][1].get_params()


            )

    assert (my_pipe.named_steps.lr
            == my_pipe['lr']
            == (my_pipe._pipeline.named_steps.lr)
            )

    with pytest.raises(ValueError, match='Indexing must be done '):
        my_pipe[0]


def test_preprocess_all_ExtendedPipeline():
    feature_steps = [
        ('zscore', StandardScaler()),
        ('pca', PCA())
    ]

    confound_steps = [('zscore', StandardScaler()),
                      ('zscore_2', StandardScaler())]

    y_transformer = TargetTransformerWrapper(StandardScaler())

    feature_pipe = Pipeline(feature_steps)
    confounds_pipe = Pipeline(confound_steps)
    extended_pipe = make_pipeline(
        feature_steps,
        y_transformer=y_transformer,
        confound_steps=confound_steps)

    np.random.seed(42)
    X_trans_preprocess = extended_pipe.fit_transform(X, y, n_confounds=1)

    np.random.seed(42)
    conf_trans = confounds_pipe.fit_transform(ensure_2d(X.C), y)
    y_trans = y_transformer.fit_transform(X, y)
    X_trans = feature_pipe.fit_transform(X[['A', 'B']], y_trans)

    X_trans_preprocess, y_trans_preprocess, conf_preprocess = \
        extended_pipe.preprocess(X, y, until='pca')

    assert_array_almost_equal(
        X_trans, X_trans_preprocess.iloc[:, :2])  # type: ignore
    assert_array_equal(conf_trans, conf_preprocess)
    assert_array_equal(y_trans, y_trans_preprocess)


def test_preprocess_column_names():
    feature_steps = [
        ('zscore', StandardScaler()),
        ('zscore2', StandardScaler()),
        ('zscore3', StandardScaler()),
        ('pca', PCA()),
    ]

    confound_steps = [('zscore', StandardScaler()),
                      ('zscore_2', StandardScaler())]

    y_transformer = TargetTransformerWrapper(StandardScaler())
    extended_pipe = make_pipeline(
        feature_steps,
        y_transformer=y_transformer,
        confound_steps=confound_steps)

    np.random.seed(42)
    X_trans_preprocess = extended_pipe.fit_transform(
        X, y, n_confounds=1)

    prep_col_names, *_ = extended_pipe.preprocess(
        X, y, column_names=X.columns.tolist())
    prep_no_col_names, *_ = extended_pipe.preprocess(X, y)

    assert_array_almost_equal(X_trans_preprocess, prep_col_names.values)
    assert_frame_equal(prep_col_names, prep_no_col_names)


def test_preprocess_until_ExtendedPipeline():
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
    extended_pipe.fit(X, y, n_confounds=1)
    np.random.seed(42)
    X_trans = X.values[:, :2]
    y_trans = y.copy()
    conf_trans = ensure_2d(X.C)

    for name, step in confound_steps:

        this_confounds_pipe = make_pipeline(steps=[(name, step)])
        conf_trans = this_confounds_pipe.fit_transform(conf_trans)

        X_trans_pipe, y_trans_pipe, conf_trans_pipe = extended_pipe.preprocess(
            X, y, until=f'confounds__{name}')

        assert_array_equal(X, X_trans_pipe)
        assert_array_equal(y, y_trans_pipe)
        assert_array_equal(conf_trans, conf_trans_pipe)

    y_trans = y_transformer.fit_transform(ensure_2d(y_trans)).squeeze()
    X_trans_pipe, y_trans_pipe, conf_trans_pipe = extended_pipe.preprocess(
        X, y, until='target__transformer',)
    assert_array_equal(
        X.iloc[:, :2], X_trans_pipe.iloc[:, :2])  # type: ignore
    assert_array_equal(y_trans, y_trans_pipe)
    assert_array_equal(conf_trans, conf_trans_pipe)

    for name, step in feature_steps:
        this_feature_pipe = make_pipeline(steps=[(name, step)])
        X_trans = this_feature_pipe.fit_transform(X_trans)
        X_trans_pipe, y_trans_pipe, conf_trans_pipe = \
            extended_pipe.preprocess(X, y, until=name)

        assert_array_almost_equal(
            X_trans, X_trans_pipe.iloc[:, :2])  # type: ignore
        assert_array_equal(y_trans, y_trans_pipe)
        assert_array_equal(conf_trans, conf_trans_pipe)

    with pytest.raises(ValueError, match='banana_pie is not a valid'):
        extended_pipe.preprocess(X, y, until='banana_pie')


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
        X.iloc[:, :-1], X.C, n_confounds=1).predict(X.iloc[:, :-1])


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
    extended_pipe.fit(X, y, n_confounds=1)
    for param, val in params.items():
        extended_pipe.set_params(**{param: val})

        assert extended_pipe.get_params()[param] == val

    with pytest.raises(ValueError, match='You cannot set'):
        extended_pipe.set_params(cOnFouunds__zscore__with_mean=True)

    with pytest.raises(ValueError, match='You cannot set'):
        extended_pipe.set_params(confounds__Isacore__with_mean=True)

    with pytest.raises(ValueError, match='You cannot set'):
        extended_pipe.set_params(target__Isacore__with_mean=True)


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


def test_set_params_errors():
    pipe = make_pipeline([('lr', LogisticRegression())])

    with pytest.raises(ValueError,
                       match='You cannot set bananapie'):
        clone(pipe).set_params(bananapie=2)

    with pytest.raises(ValueError,
                       match='You cannot set parameters for the confound'):
        clone(pipe).set_params(confounds__not_valid=2)

    with pytest.raises(AttributeError,
                       match='Your y_transformer seems to be None '):
        clone(pipe).set_params(target__not_valid=2)

    pipe.fit(X, y_bin)
    with pytest.raises(AttributeError,
                       match='Your confounding pipeline seems to be None'):
        pipe.set_params(confounds__not_valid=2)

    with pytest.raises(AttributeError,
                       match='Your y_transformer seems to be None '):
        pipe.set_params(target__not_valid=2)


def test_nested_hyperparameters():
    steps = [
        ('trans', ColumnTransformer(
            transformers=[('zscore', StandardScaler(), slice(None, -1))],
            remainder='passthrough')),
        ('lr', LogisticRegression())
    ]
    confound_steps = [
        ('trans', ColumnTransformer(
            transformers=[('zscore', StandardScaler(), slice(None, -1))],
            remainder='passthrough'))
    ]
    pipe = make_pipeline(
        steps=steps,
        confound_steps=confound_steps

    )
    pipe.set_params(trans__zscore__with_mean=False)
    pipe.set_params(confounds__trans__zscore__with_mean=False)

    pipe.fit(X_with_types, y_bin, n_confounds=2)
