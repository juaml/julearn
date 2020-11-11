# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from julearn.transformers import DataFrameTransformer, TargetTransfromerWrapper
from julearn.pipeline import (ExtendedDataFramePipeline,
                              create_dataframe_pipeline,
                              create_extended_pipeline)

X = pd.DataFrame(dict(A=np.arange(10),
                      B=np.arange(10, 20),
                      C=np.arange(30, 40)
                      ))


def test_create_dataframe_pipeline_all_steps_added():
    steps = [('zscore', StandardScaler(), 'same'),
             ('pca', PCA(), 'unknown'),
             ('lr', LinearRegression())]
    pipe = create_dataframe_pipeline(steps)
    isinstance(pipe.steps[0], DataFrameTransformer)
    isinstance(pipe.steps[1], DataFrameTransformer)
    isinstance(pipe.steps[2], LinearRegression)


def test_create_dataframe_pipeline_steps_added_correctly():
    # test whether the steps are added
    # and whether all the hyperparameters were transfered correctly
    scaler = StandardScaler(with_mean=False)
    pca = PCA(n_components=3)
    lr = LinearRegression()
    steps = [('zscore', scaler),
             ('pca', pca),
             ('linear_reg', lr)]

    my_pipeline = create_dataframe_pipeline(steps)

    for my_step, original_estimator in zip(
            my_pipeline.steps, [scaler, pca, lr]):
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_step[1].get_params().get(est_param) == est_val


def test_create_dataframe_pipeline_returned_features_same():

    steps = [('zscore', StandardScaler(), 'same')]
    sklearn_pipe = make_pipeline(StandardScaler())

    my_pipe = create_dataframe_pipeline(steps)
    X_trans = my_pipe.fit_transform(X)
    X_trans_sklearn = sklearn_pipe.fit_transform(X)

    assert (X_trans.columns == X.columns).all()
    assert_array_equal(X_trans.values, X_trans_sklearn)


def test_ExtendedDataFramePipeline_basics_Xpipeline_no_error():
    steps = [('zscore', StandardScaler(), 'same'),
             ('pca', PCA(), 'unknown'),
             ('lr', LinearRegression())
             ]

    my_pipe = create_dataframe_pipeline(steps)
    my_transformer_pipe = create_dataframe_pipeline(steps[:-1])
    my_pipe = ExtendedDataFramePipeline(my_pipe)
    my_transform_pipe = ExtendedDataFramePipeline(my_transformer_pipe)
    my_pipe.fit(X.iloc[:, :-1], X.C)
    my_pipe.score(X.iloc[:, :-1], X.C)

    my_transform_pipe.fit_transform(X)


def test_ExtendedDataFramePipeline_basics_conf_df_pipe_no_error():
    steps = [('zscore', StandardScaler(), 'same'),
             ('pca', PCA(), 'unknown'),
             ('lr', LinearRegression())
             ]

    my_pipe = create_dataframe_pipeline(steps)
    my_confound_pipe = create_dataframe_pipeline(
        steps[0:1],
        default_returned_features='same',
        default_transform_column='confound')

    my_pipe = ExtendedDataFramePipeline(
        my_pipe,
        confound_dataframe_pipeline=my_confound_pipe, confounds=['B'])

    my_pipe.fit(X.iloc[:, :-1], X.C)
    score = my_pipe.score(X.iloc[:, :-1], X.C)
    assert score is not np.nan


def test_ExtendedDataFramePipeline_basics_y_transformer_no_error():
    steps = [('zscore', StandardScaler(), 'same'),
             ('pca', PCA(), 'unknown'),
             ('lr', LinearRegression())
             ]

    my_pipe = create_dataframe_pipeline(steps)
    y_transformer = TargetTransfromerWrapper(StandardScaler())

    my_pipe = ExtendedDataFramePipeline(
        my_pipe, y_transformer=y_transformer)

    my_pipe.fit(X.iloc[:, :-1], X.C)
    my_pipe.score(X.iloc[:, :-1], X.C)


def test_ExtendedDataFramePipeline_transform_with_categorical():

    steps = [('zscore', StandardScaler(), 'same')]
    sklearn_pipe = make_pipeline(StandardScaler())

    my_pipe = create_dataframe_pipeline(steps)
    my_pipe = ExtendedDataFramePipeline(my_pipe, categorical_features=['C'])
    X_trans = my_pipe.fit_transform(X)
    X_trans_sklearn = sklearn_pipe.fit_transform(X.loc[:, ['A', 'B']])

    assert_array_equal(X_trans.loc[:, ['A', 'B']].values, X_trans_sklearn)
    assert_frame_equal(X_trans.loc[:, ['C']], X.loc[:, ['C']])


def test_ExtendedDataFramePipeline_in_cv_no_error():

    steps = [('zscore', StandardScaler(), 'same'),
             ('pca', PCA(), 'unknown'),
             ('lr', LinearRegression())
             ]

    my_pipe = create_dataframe_pipeline(steps)
    my_confound_pipe = create_dataframe_pipeline(
        steps[0:1],
        default_returned_features='same',
        default_transform_column='confound')
    y_transformer = TargetTransfromerWrapper(StandardScaler())

    my_pipe = ExtendedDataFramePipeline(
        my_pipe, y_transformer=y_transformer,
        confound_dataframe_pipeline=my_confound_pipe, confounds=['B'])

    cross_validate(my_pipe, X.iloc[:, :-1], X.C)


def test_ExtendedDataFramePipeline_with_confound_in_cv_no_error():

    steps = [('zscore', StandardScaler(), 'same'),
             ('pca', PCA(), 'unknown'),
             ('lr', LinearRegression())
             ]

    my_pipe = create_dataframe_pipeline(steps)
    my_confound_pipe = create_dataframe_pipeline(
        steps[0:1],
        default_returned_features='same',
        default_transform_column='confound')
    y_transformer = TargetTransfromerWrapper(StandardScaler())

    my_pipe = ExtendedDataFramePipeline(
        my_pipe, y_transformer=y_transformer,
        confound_dataframe_pipeline=my_confound_pipe,
        confounds='B')

    cross_validate(my_pipe, X.iloc[:, :-1], X.C)


def test_create_extended_dataframe_transformer():
    preprocess_steps_feature = [('zscore', StandardScaler(), 'same'),
                                ('pca', PCA(), 'unknown')
                                ]

    model = ('lr', LinearRegression())
    conf_steps = [('zscore', StandardScaler())]
    y_transformer = TargetTransfromerWrapper(StandardScaler())
    extended_pipe = create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_feature,
        preprocess_transformer_target=y_transformer,
        preprocess_steps_confounds=conf_steps,
        model=model,
        confounds='B',
        categorical_features=None
    )

    extended_pipe.fit(X.iloc[:, :-1], X.C)
    extended_pipe.predict(X.iloc[:, :-1])
    score = extended_pipe.score(X.iloc[:, :-1], X.C)
    assert score is not np.nan


def test_access_steps_ExtendedDataFramePipeline():
    steps = [('zscore', StandardScaler(), 'same'),
             ('pca', PCA(), 'unknown'),
             ('lr', LinearRegression())
             ]

    my_pipe = create_dataframe_pipeline(steps)
    my_confound_pipe = create_dataframe_pipeline(
        steps[0:1],
        default_returned_features='same',
        default_transform_column='confound')
    y_transformer = TargetTransfromerWrapper(StandardScaler())

    my_pipe = ExtendedDataFramePipeline(
        my_pipe, y_transformer=y_transformer,
        confound_dataframe_pipeline=my_confound_pipe, confounds=['B'])

    assert (my_pipe.named_confound_steps.zscore
            == my_pipe['confound_zscore']
            == (my_pipe
                .confound_dataframe_pipeline
                .named_steps
                .zscore)
            )

    assert (my_pipe.named_steps.pca
            == my_pipe['pca']
            == (my_pipe
                .dataframe_pipeline
                .named_steps
                .pca)
            )

    assert (my_pipe.named_steps.lr
            == my_pipe['lr']
            == (my_pipe
                .dataframe_pipeline
                .named_steps
                .lr)
            )

    with pytest.raises(ValueError, match='Indexing must be done '):
        my_pipe[0]
