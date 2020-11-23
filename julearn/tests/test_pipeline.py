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

from julearn.transformers import (DataFrameTransformer,
                                  TargetTransfromerWrapper,
                                  DataFrameConfoundRemover)
from julearn.pipeline import (ExtendedDataFramePipeline,
                              create_dataframe_pipeline,
                              create_extended_pipeline)
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
    # and whether all the hyperparameters were transferred correctly
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


def test_create_dataframe_pipeline_invalid_step():

    steps = [('zscore',)]
    with pytest.raises(ValueError, match='step:'):
        create_dataframe_pipeline(steps)


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
    assert_array_equal(X_trans.loc[:, ['C__:type:__categorical']].values,
                       X.loc[:, ['C']].values)


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


def test_preprocess_all_ExtendedDataFramePipeline():
    feature_steps = [('zscore', StandardScaler(), 'same'),
                     ('pca', PCA(), 'unknown'),
                     ]
    model = ('lr', LinearRegression())

    steps = feature_steps + [model]
    confound_steps = [('zscore', StandardScaler(), 'same'),
                      ('zscore_2', StandardScaler(), 'same')]

    y_transformer = TargetTransfromerWrapper(StandardScaler())

    feature_pipe = create_dataframe_pipeline(steps=feature_steps)
    steps_pipe = create_dataframe_pipeline(steps=steps)
    confounds_pipe = create_dataframe_pipeline(
        steps=confound_steps,
        default_returned_features='same',
        default_transform_column='confound')

    extended_pipe = ExtendedDataFramePipeline(
        dataframe_pipeline=steps_pipe,
        y_transformer=y_transformer,
        confound_dataframe_pipeline=confounds_pipe,
        confounds=['B'])

    np.random.seed(42)
    extended_pipe.fit(X, y)

    np.random.seed(42)
    X_recoded = extended_pipe._recode_columns(X.copy())
    X_conf = confounds_pipe.fit_transform(X_recoded, y)
    y_trans = y_transformer.fit_transform(X_conf, y)
    X_trans = feature_pipe.fit_transform(X_conf, y_trans)

    X_trans_preprocess, y_trans_preprocess = extended_pipe.preprocess(
        X, y, return_trans_column_type=True)

    X_trans_preprocess = extended_pipe._recode_columns(X_trans_preprocess)
    assert_frame_equal(X_trans, X_trans_preprocess)
    assert_array_equal(y_trans, y_trans_preprocess)


def test_preprocess_until_ExtendedDataFramePipeline():
    feature_steps = [('zscore', StandardScaler(), 'same'),
                     ('pca', PCA(), 'unknown'),
                     ]
    model = ('lr', LinearRegression())

    steps = feature_steps + [model]
    confound_steps = [('zscore', StandardScaler(), 'same'),
                      ('zscore_2', StandardScaler(), 'same')]

    y_transformer = TargetTransfromerWrapper(StandardScaler())

    steps_pipe = create_dataframe_pipeline(steps=steps)
    confounds_pipe = create_dataframe_pipeline(
        steps=confound_steps,
        default_returned_features='same',
        default_transform_column='confound')

    extended_pipe = ExtendedDataFramePipeline(
        dataframe_pipeline=steps_pipe,
        y_transformer=y_transformer,
        confound_dataframe_pipeline=confounds_pipe,
        confounds=['B'])

    np.random.seed(42)
    extended_pipe.fit(X, y)

    np.random.seed(42)
    X_trans = extended_pipe._recode_columns(X.copy())
    y_trans = y.copy()

    for name, step, returns in confound_steps:

        this_confounds_pipe = create_dataframe_pipeline(
            steps=[(name, step, returns)],
            default_returned_features='same',
            default_transform_column='confound')
        X_trans = this_confounds_pipe.fit_transform(X_trans)

        X_trans_pipe, y_trans_pipe = extended_pipe.preprocess(
            X, y, until='confound_' + name, return_trans_column_type=True)

        assert_frame_equal(X_trans, X_trans_pipe)
        assert_array_equal(y_trans, y_trans_pipe)

    X_trans_pipe, y_trans_pipe = extended_pipe.preprocess(
        X, y, until='target_', return_trans_column_type=True)
    y_trans = y_transformer.fit_transform(X_trans, y_trans)
    assert_frame_equal(X_trans, X_trans_pipe)
    assert_array_equal(y_trans, y_trans_pipe)

    for name, step, returns in feature_steps:

        this_feature_pipe = create_dataframe_pipeline(
            steps=[(name, step, returns)])
        X_trans = this_feature_pipe.fit_transform(X_trans)

        X_trans_pipe, y_trans_pipe = extended_pipe.preprocess(
            X, y, until=name, return_trans_column_type=True)

        X_trans_pipe_types, y_trans_pipe_types = extended_pipe.preprocess(
            X, y, until=name, return_trans_column_type=False)

        assert_frame_equal(X_trans, X_trans_pipe)
        assert_array_equal(y_trans, y_trans_pipe)

        assert_array_equal(X_trans_pipe_types.values,
                           X_trans_pipe_types.values)
        assert_array_equal(y_trans_pipe_types, y_trans_pipe_types)

    with pytest.raises(ValueError, match='banana_pie is not a valid'):
        extended_pipe.preprocess(
            X, y, 'banana_pie', return_trans_column_type=True)


def test_remove_column_types_ExtendedDataFramePipe():

    feature_steps = [('zscore', StandardScaler(), 'same'),
                     ('pca', PCA(), 'unknown'),
                     ]
    model = ('lr', LinearRegression())

    steps = feature_steps + [model]
    confound_steps = [('zscore', StandardScaler(), 'same'),
                      ('zscore_2', StandardScaler(), 'same')]

    y_transformer = TargetTransfromerWrapper(StandardScaler())

    steps_pipe = create_dataframe_pipeline(steps=steps)
    confounds_pipe = create_dataframe_pipeline(
        steps=confound_steps,
        default_returned_features='same',
        default_transform_column='confound')

    extended_pipe = ExtendedDataFramePipeline(
        dataframe_pipeline=steps_pipe,
        y_transformer=y_transformer,
        confound_dataframe_pipeline=confounds_pipe,
        confounds=['B'])

    extended_pipe.fit(X, y)
    X_removed = extended_pipe._remove_column_types(X_with_types)
    assert (X_removed.columns == list('abcdef')).all()


def test_create_exteneded_pipeline_confound_removal():
    '''Test automatic drop of confounds at the end of the pipeline
    '''
    preprocess_steps_feature = [(
        'remove_confound',
        DataFrameConfoundRemover(),
        'subset', 'all'),
    ]
    preprocess_steps_feature_keep_conf = [(
        'remove_confound',
        DataFrameConfoundRemover(keep_confounds=True),
        'subset', 'all'),
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

    extended_pipe_keep_confound = create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_feature_keep_conf,
        preprocess_transformer_target=y_transformer,
        preprocess_steps_confounds=conf_steps,
        model=model,
        confounds='B',
        categorical_features=None
    )
    np.random.seed(4242)
    pred = (extended_pipe
            .fit(X.iloc[:, :-1], X.C)
            .predict(X.iloc[:, :-1]))

    np.random.seed(4242)
    pred_keep = (extended_pipe_keep_confound
                 .fit(X.iloc[:, :-1], X.C)
                 .predict(X.iloc[:, :-1]))

    assert_array_equal(pred, pred_keep)
