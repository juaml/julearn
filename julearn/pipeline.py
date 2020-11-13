# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone

from . transformers.dataframe import DataFrameTransformer
from . utils import raise_error


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
            raise_error(f'step: {i_step} has a len of {len(i_step)}'
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
        convert most sklearn transformers to a target_transformer.
    confound_dataframe_pipeline : sklearn.pipeline.Pipeline, optional
        Similar to dataframe_pipeline.
    confounds : list[str], optional
        a list of column names which are confounds ,by default None
    categorical_features : list[str], optional
        a list of column names which are cateroical features,by default None
    return_trans_column_type : bool, optional
        whether to return transformed column names with the associated
        column type, by default False
    """

    column_type_sep = '__:type:__'

    def __init__(self, dataframe_pipeline,
                 y_transformer=None,
                 confound_dataframe_pipeline=None,
                 confounds=None, categorical_features=None,
                 return_trans_column_type=False):

        self.dataframe_pipeline = dataframe_pipeline
        self.y_transformer = y_transformer
        self.confound_dataframe_pipeline = confound_dataframe_pipeline
        self.confounds = confounds
        self.categorical_features = categorical_features
        self.return_trans_column_type = return_trans_column_type

    def fit(self, X, y=None):

        self.dataframe_pipeline = clone(self.dataframe_pipeline)
        self.confound_dataframe_pipeline = (
            None
            if self.confound_dataframe_pipeline is None
            else clone(self.confound_dataframe_pipeline)
        )
        self.y_transformer = (
            None
            if self.y_transformer is None
            else clone(self.y_transformer)
        )

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

    def preprocess(self, X, y, until=None):
        # TODO incase no model at the end
        old_model = self.dataframe_pipeline.steps.pop()
        if until is None:
            X_trans = self.transform(X)
            y_trans = self.transform_target(X, y)
        else:
            if self[until] is None:
                raise_error(f'{until} is not a valid step')
            elif until.startswith('confound_'):
                step_name = until.replace('confound_', '')
                X_trans = self._transform_pipeline_until(
                    pipeline=self.confound_dataframe_pipeline,
                    step_name=step_name,
                    X=X
                )
                y_trans = y.copy()

            elif until.startswith('target_'):
                X_trans = self.transform_confounds(X)
                y_trans = self.transform_target(X, y)
            else:
                X_trans = self.transform_confounds(X)
                X_trans = self._transform_pipeline_until(
                    pipeline=self.dataframe_pipeline,
                    step_name=until,
                    X=X_trans
                )
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
            raise_error('Indexing must be done using strings')
        if ind.startswith('confound_'):
            n_ind = ind.replace('confound_', '')
            element = self.confound_dataframe_pipeline[n_ind]
        elif ind.startswith('target_'):
            element = self.y_transformer
        else:
            element = self.dataframe_pipeline[ind]
        return element

    def _transform_pipeline_until(self, pipeline, step_name, X):
        X_transformed = X.copy()
        X_transformed = self._recode_columns(X_transformed)
        for name, step in pipeline.steps:
            X_transformed = step.transform(X_transformed)
            if name == step_name:
                break
        return X_transformed

    @property
    def named_steps(self):
        return self.dataframe_pipeline.named_steps

    @property
    def named_confound_steps(self):
        return self.confound_dataframe_pipeline.named_steps


def create_extended_pipeline(
    preprocess_steps_features,
    preprocess_transformer_target,
    preprocess_steps_confounds,
    model, confounds, categorical_features
):
    """

    Parameters
    ----------
    preprocess_steps_feature: list[tuple]
        Is a list of tuples. Each tuple can contain 2-4 entries.
        The first is always the name of the step as a str.
        Second the model/transformer following sklearns style.
        Third (optional) returned_features following DataFrameTransformer.
        Fourth (optional) transform_column follwing DataFrameTransformer.

    preprocess_transformer_target : y_transform
        A transformer, which takes in X, y and outputs a transformed y.

    preprocess_steps_confounds : list[tuple]
        Is a list of tuples. Each tuple can contain 2-3 entries.
        The first is always the name of the step as a str.
        Second the transformer following sklearns style.
        Third (optional) returned_features following DataFrameTransformer,
        but they should only be `same` (default) or `unknown_same_type`.
    model : tuple(str, sklearn.base.BaseEstimator)

    confounds : list[str] or str
        A list of column_names which are the confounds
        or the column_name of one confound

    categorical_features : list[str] or str
        A list of column_names which are the categorical features
        or the column_name of one categorical feature
    """
    X_steps = (list(preprocess_steps_features) + [model]
               if preprocess_steps_features is not None
               else [model]
               )
    pipeline = create_dataframe_pipeline(X_steps)

    if preprocess_steps_confounds is not None:
        confound_pipe = create_dataframe_pipeline(
            preprocess_steps_confounds, default_returned_features='same',
            default_transform_column='confound')
    else:
        confound_pipe = None

    return ExtendedDataFramePipeline(
        dataframe_pipeline=pipeline,
        y_transformer=preprocess_transformer_target,
        confound_dataframe_pipeline=confound_pipe,
        confounds=confounds,
        categorical_features=categorical_features)
