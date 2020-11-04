from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from . transformers.dataframe import DataFrameTransformer


def create_pipeline(preprocess_steps_features,
                    preprocess_transformer_target,
                    preprocess_steps_confounds,
                    model_tuple, confounds, categorical_features):

    X_steps = list(preprocess_steps_features) + [model_tuple]
    y_transformer = preprocess_transformer_target
    pipeline = create_extended_pipeline(
        X_steps, y_transformer, preprocess_steps_confounds, confounds,
        categorical_features)
    return pipeline


def create_dataframe_pipeline(steps,
                              default_returned_features='unknown',
                              default_transform_column='continuous'):
    """Creates a Sklearn pipeline using the provided steps and wrapping all
    transformers into the DataFrameTransformer.

    Parameters
    ----------
    steps : list[tuple, tuple ...]
        Are a list of tuples. Each tuple can contain 2-4 entries.
        The first is always the name of the step as a str.
        Second the model/transformer following sklearns style.
        Third (optional) returned_features following DataFrameTransformer.
        Fourth (optional) transform_column follwing DataFrameTransformer.
        The last tuple can be a tuple of (model_name, model).
    default_returned_features : str, optional
        When a step does not provide a returned_features/third entry,
        this will provide the default for it.
    default_transform_column : str, optional
        When a step does not provide a returned_features/Fourth entry,
        this will provide the default for it.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A Pipline in which all entries are wrappend in a DataFrameTransformer.
    """
    steps_ready_for_pipe = []
    for i_step, step in enumerate(steps):

        n_arguments = len(step)
        returned_features = default_returned_features
        transform_column = default_transform_column
        if n_arguments == 2:
            name, estimator = step
        elif n_arguments == 3:
            name, estimator, returned_features = step
        elif n_arguments == 4:
            name, estimator, returned_features, transform_column = step

        else:
            raise ValueError(f'step: {i_step} has a len of {len(i_step)}'
                             ', but should hve one between 2 and 4')

        if (i_step == len(steps) - 1) and (hasattr(estimator, 'predict')):
            steps_ready_for_pipe.append([name, estimator])
        else:
            transformer = DataFrameTransformer(
                transformer=estimator,
                transform_column=transform_column,
                returned_features=returned_features)
            steps_ready_for_pipe.append([name, transformer])

    return Pipeline(steps=steps_ready_for_pipe)


class ExtendedDataFramePipeline(BaseEstimator):
    """A class creating a custom metamodel like a Pipeline.
    In practice this should be created
    using julearn.pipeline.create_extended_pipeline.
    Added functionality:
    1: handling target transforming and scoring against
    this transformed target as ground truth.
    2: handling confounds. Adds the confound as type to columns.
    This allows the DataFrameTransformer inside of the dataframe_pipeline
    to handle confounds properly.
    3: Handling categorical features:
    Adds categorical type to columns so that DataFrameTransformer inside
    of the dataframe_pipline can handle categorical features properly.

    column_types are added to the feature dataframe after each column using
    the specified seperater. E.g. column `cheese` becomes
    `cheese__:type:__confound`, when being in the confounds
    and with a seperater of `__:type:__`


    dataframe_pipeline : sklearn.pipeline.Pipeline
        A pipeline working with dataframes and being able to handle confounds.
        Should be created using julearn.pipeline.create_dataframe_pipeline.
    y_transformer : julearn target_transformer, optional
        Any transformer which can take the X and y to transform the y.
        You can use julearn.transformers.target.TargetTransfromerWrapper to
        convert most sklearn transformers to a target_transformer
        , by default TargetPassThroughTransformer()
    confound_dataframe_pipeline : sklearn.pipeline.Pipeline, optional
        Similar to dataframe_pipeline, by default PassThroughTransformer()
    confounds : list[str], optional
        a list of column names which are confounds ,by default None
    categorical_features : list[str], optional
        a list of column names which are cateroical features,by default None
    column_type_sep : str, optional
        this seperater is used to change the column names internally and add
        a column type after the actual column name
        seperated by the column_type_sep, by default '__:type:__'
    return_trans_column_type : bool, optional
        whether to return transformed column names with the associated
        column type, by default False
    """

    def __init__(self, dataframe_pipeline,
                 y_transformer=None,
                 confound_dataframe_pipeline=None,
                 confounds=None, categorical_features=None,
                 column_type_sep='__:type:__',
                 return_trans_column_type=False):

        self.dataframe_pipeline = dataframe_pipeline
        self.y_transformer = y_transformer
        self.confound_dataframe_pipeline = confound_dataframe_pipeline
        self.confounds = confounds
        self.categorical_features = categorical_features
        self.column_type_sep = column_type_sep
        self.return_trans_column_type = return_trans_column_type

    def fit(self, X, y=None):
        if self.categorical_features is None:
            self.categorical_features = []

        self._set_column_mappers(X)

        X = self._recode_columns(X)

        if self.confound_dataframe_pipeline is not None:
            X_conf_trans = self._fit_transform_confounds(X, y)
        else:
            X_conf_trans = X

        if self.y_transformer is not None:
            y_true = self.y_transformer.fit_transform(X_conf_trans, y)
        else:
            y_true = y

        self.dataframe_pipeline.fit(X_conf_trans, y_true)

        return self

    def predict(self, X):
        X_conf_trans = self.transform_confounds(X)
        return self.dataframe_pipeline.predict(X_conf_trans)

    def predict_proba(self, X):
        X_conf_trans = self.transform_confounds(X)
        return self.dataframe_pipeline.predict_proba(X_conf_trans)

    def score(self, X, y):
        X_conf_trans = self.transform_confounds(X)
        y_true = self.transform_target(X, y)
        return self.dataframe_pipeline.score(X_conf_trans, y_true)

    def transform(self, X):
        X_conf_trans = self.transform_confounds(X)
        X_trans = self.dataframe_pipeline.transform(X_conf_trans)
        if not self.return_trans_column_type:
            X_trans = (X_trans
                       .rename(columns=self.col_name_mapper_inverse_)
                       .copy()
                       )
        return X_trans

    def transform_target(self, X, y):
        X_conf_trans = self.transform_confounds(X)
        y_true = self._transform_target(X_conf_trans, y)
        return y_true

    def transform_confounds(self, X):
        X = self._recode_columns(X)
        if self.confounds is None:
            return X
        else:
            return self.confound_dataframe_pipeline.transform(X)

    def preprocess(self, X, y):
        old_model = self.dataframe_pipeline.steps.pop()
        X_trans = self.transform(X, y)
        y_trans = self.transform_target(X, y)
        self.dataframe_pipeline.steps.append(old_model)
        return X_trans, y_trans

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _fit_transform_confounds(self, X, y):
        X = self._recode_columns(X)
        if self.confounds is None:
            return X
        else:
            return self.confound_dataframe_pipeline.fit_transform(X, y)

    def _transform_target(self, X, y):
        if self.y_transformer is not None:
            y_true = self.y_transformer.transform(X, y)
        else:
            y_true = y
        return y_true

    def _convert_column_name(self, col_name):
        confounds = [] if self.confounds is None else self.confounds

        if col_name in confounds:
            out_col_name = col_name + self.column_type_sep + 'confound'
        elif col_name in self.categorical_features:
            out_col_name = col_name + self.column_type_sep + 'categorical'
        else:
            out_col_name = col_name

        return out_col_name

    def _set_column_mappers(self, X):
        self.col_name_mapper_ = {col_name: self._convert_column_name(col_name)
                                 for col_name in X.columns
                                 }

        self.col_name_mapper_inverse_ = {
            val: key
            for key, val in self.col_name_mapper_.items()

        }

    def _recode_columns(self, X):
        return X.rename(columns=self.col_name_mapper_).copy()

    def __getitem__(self, ind):
        if not isinstance(ind, str):
            raise ValueError('Indexing must be done using strings')
        if ind.startswith('confound_'):
            n_ind = ind.replace('confound_', '')
            element = self.confound_dataframe_pipeline[n_ind]
        else:
            element = self.dataframe_pipeline[ind]
        return element

    @property
    def named_steps(self):
        return self.dataframe_pipeline.named_steps

    @property
    def named_confound_steps(self):
        return self.confound_dataframe_pipeline.named_steps


def create_extended_pipeline(X_steps, y_transformer, conf_steps,
                             confounds, categorical_features):
    """[summary]

    Parameters
    ----------
    X_steps : list[tuple]
        Is a list of tuples. Each tuple can contain 2-4 entries.
        The first is always the name of the step as a str.
        Second the model/transformer following sklearns style.
        Third (optional) returned_features following DataFrameTransformer.
        Fourth (optional) transform_column follwing DataFrameTransformer.
        The last tuple can be a tuple of (model_name, model).

    y_transformer : y_transform
        A transformer, which takes in X, y and outputs a transformed y.

    conf_steps : list[tuple]
        Is a list of tuples. Each tuple can contain 2-3 entries.
        The first is always the name of the step as a str.
        Second the transformer following sklearns style.
        Third (optional) returned_features following DataFrameTransformer,
        but they should only be `same` (default) or `unknown_same_type`.

    confounds : list[str] or str
        A list of column_names which are the confounds
        or the column_name of one confound

    categorical_features : list[str] or str
        A list of column_names which are the categorical features
        or the column_name of one categorical feature
    """
    pipeline = create_dataframe_pipeline(X_steps)

    # TODO validate conf_steps to only be unknown_same_type or same
    # and not 4 entries
    confound_pipe = create_dataframe_pipeline(
        conf_steps, default_returned_features='same',
        default_transform_column='confound')

    return ExtendedDataFramePipeline(dataframe_pipeline=pipeline,
                                     y_transformer=y_transformer,
                                     confound_dataframe_pipeline=confound_pipe,
                                     confounds=confounds,
                                     categorical_features=categorical_features)
