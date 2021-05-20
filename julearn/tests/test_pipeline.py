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
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from julearn.transformers import (
    TargetTransfromerWrapper, DataFrameConfoundRemover, get_transformer
)
from julearn.pipeline import (ExtendedDataFramePipeline,
                              create_dataframe_pipeline,
                              _create_extended_pipeline)
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

    steps = [('zscore', StandardScaler())]
    sklearn_pipe = make_pipeline(StandardScaler())

    my_pipe = create_dataframe_pipeline(steps)
    X_trans = my_pipe.fit_transform(X)
    X_trans_sklearn = sklearn_pipe.fit_transform(X)

    assert (X_trans.columns == X.columns).all()
    assert_array_equal(X_trans.values, X_trans_sklearn)


def test_ExtendedDataFramePipeline_transform_with_categorical():

    steps = [('zscore', StandardScaler())]
    sklearn_pipe = make_pipeline(StandardScaler())

    my_pipe = create_dataframe_pipeline(steps)
    my_pipe = ExtendedDataFramePipeline(my_pipe, categorical_features=['C'])
    X_trans = my_pipe.fit_transform(X)
    X_trans_sklearn = sklearn_pipe.fit_transform(X.loc[:, ['A', 'B']])

    assert_array_equal(X_trans.loc[:, ['A', 'B']].values, X_trans_sklearn)
    assert_array_equal(X_trans.loc[:, ['C__:type:__categorical']].values,
                       X.loc[:, ['C']].values)


def test_create_extended_dataframe_transformer():
    preprocess_steps_feature = [('zscore', StandardScaler()),
                                ('pca', PCA())
                                ]

    model = ('lr', LinearRegression())
    conf_steps = [('zscore', StandardScaler())]
    y_transformer = TargetTransfromerWrapper(StandardScaler())
    extended_pipe = _create_extended_pipeline(
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
    steps = [('zscore', StandardScaler()),
             ('pca', PCA()),
             ('lr', LinearRegression())
             ]

    my_pipe = create_dataframe_pipeline(steps)
    my_confound_pipe = create_dataframe_pipeline(
        steps[0:1],
        apply_to='confound')
    y_transformer = TargetTransfromerWrapper(StandardScaler())

    my_pipe = ExtendedDataFramePipeline(
        my_pipe, y_transformer=y_transformer,
        confound_dataframe_pipeline=my_confound_pipe, confounds=['B'])

    assert (my_pipe.named_confound_steps.zscore
            == my_pipe['confound__zscore']
            == (my_pipe
                .confound_dataframe_pipeline
                .named_steps
                .zscore
                .transformer
                )
            )

    assert (my_pipe.named_steps.pca
            == my_pipe['pca']
            == (my_pipe
                .dataframe_pipeline
                .named_steps
                .pca
                .transformer
                )
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
    feature_steps = [('zscore', StandardScaler()),
                     ('pca', PCA()),
                     ]
    model = ('lr', LinearRegression())

    steps = feature_steps + [model]
    confound_steps = [('zscore', StandardScaler()),
                      ('zscore_2', StandardScaler())]

    y_transformer = TargetTransfromerWrapper(StandardScaler())

    feature_pipe = create_dataframe_pipeline(steps=feature_steps)
    steps_pipe = create_dataframe_pipeline(steps=steps)
    confounds_pipe = create_dataframe_pipeline(
        steps=confound_steps,
        apply_to='confound')

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
    feature_steps = [('zscore', StandardScaler()),
                     ('pca', PCA()),
                     ]
    model = ('lr', LinearRegression())

    steps = feature_steps + [model]
    confound_steps = [('zscore', StandardScaler()),
                      ('zscore_2', StandardScaler())]

    y_transformer = TargetTransfromerWrapper(StandardScaler())

    steps_pipe = create_dataframe_pipeline(steps=steps)
    confounds_pipe = create_dataframe_pipeline(
        steps=confound_steps,
        apply_to='confound')

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

    for name, step in confound_steps:

        this_confounds_pipe = create_dataframe_pipeline(
            steps=[(name, step)],
            apply_to='confound'
        )
        X_trans = this_confounds_pipe.fit_transform(X_trans)

        X_trans_pipe, y_trans_pipe = extended_pipe.preprocess(
            X, y, until='confound__' + name, return_trans_column_type=True)

        assert_frame_equal(X_trans, X_trans_pipe)
        assert_array_equal(y_trans, y_trans_pipe)

    X_trans_pipe, y_trans_pipe = extended_pipe.preprocess(
        X, y, until='target__', return_trans_column_type=True)
    y_trans = y_transformer.fit_transform(X_trans, y_trans)
    assert_frame_equal(X_trans, X_trans_pipe)
    assert_array_equal(y_trans, y_trans_pipe)

    for name, step in feature_steps:

        this_feature_pipe = create_dataframe_pipeline(
            steps=[(name, step)])
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

    feature_steps = [('zscore', StandardScaler()),
                     ('pca', PCA()),
                     ]
    model = ('lr', LinearRegression())

    steps = feature_steps + [model]
    confound_steps = [('zscore', StandardScaler()),
                      ('zscore_2', StandardScaler())]

    y_transformer = TargetTransfromerWrapper(StandardScaler())

    steps_pipe = create_dataframe_pipeline(steps=steps)
    confounds_pipe = create_dataframe_pipeline(
        steps=confound_steps,
        apply_to='confound')

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
        DataFrameConfoundRemover()),
    ]
    preprocess_steps_feature_keep_conf = [(
        'remove_confound',
        DataFrameConfoundRemover(keep_confounds=True)),
    ]

    model = ('lr', LinearRegression())
    conf_steps = [('zscore', StandardScaler())]
    y_transformer = TargetTransfromerWrapper(StandardScaler())
    extended_pipe = _create_extended_pipeline(
        preprocess_steps_features=preprocess_steps_feature,
        preprocess_transformer_target=y_transformer,
        preprocess_steps_confounds=conf_steps,
        model=model,
        confounds='B',
        categorical_features=None
    )

    extended_pipe_keep_confound = _create_extended_pipeline(
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


def test_tune_params():
    params = {'svm__kernel': 'linear',
              'zscore__with_mean': True,
              'confounds__zscore__with_mean': True,
              'target__with_mean': True}

    extended_pipe = _create_extended_pipeline(
        preprocess_steps_features=[('zscore', get_transformer('zscore'))],
        preprocess_steps_confounds=[('zscore', get_transformer('zscore'))],
        preprocess_transformer_target=get_transformer('zscore', target=True),
        model=('svm', SVR()),
        confounds=None,
        categorical_features=None
    )

    extended_pipe.set_params(**params)
    for param, val in params.items():
        assert extended_pipe.get_params()[param] == val

    with pytest.raises(ValueError, match='Each element of the'):
        extended_pipe.set_params(cOnFouunds__zscore__with_mean=True)


def test_ExtendedDataFramePipeline___rpr__():
    extended_pipe = _create_extended_pipeline(
        preprocess_steps_features=[('zscore', get_transformer('zscore'))],
        preprocess_steps_confounds=[('zscore', get_transformer('zscore'))],
        preprocess_transformer_target=get_transformer('zscore', target=True),
        model=('svm', SVR()),
        confounds=None,
        categorical_features=None
    )
    extended_pipe.__repr__()


def test_extended_pipeline_get_wrapped_transformer_params():
    steps = [('zscore', StandardScaler(with_mean=False))]

    my_pipe = create_dataframe_pipeline(steps)
    extended_pipe = ExtendedDataFramePipeline(my_pipe)
    extended_pipe.fit(X)
    assert extended_pipe['zscore'].with_mean is False
