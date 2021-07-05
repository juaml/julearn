# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Bunch

from . transformers import DataFrameWrapTransformer, DropColumns
from . utils import raise_error


def create_dataframe_pipeline(steps, apply_to=None):
    """Creates a sklearn pipeline using the provided steps and wrapping all
    transformers into the DataFrameWrapTransformer.

    Parameters
    ----------
    steps : list
        A list of steps. Each step is a tuple containing a name and transformer
        or estimator. For more information look at steps in:
        sklearn.pipeline.Pipeline
    apply_to : str | list(str) or None
        decides which columns will be transformed.
        For more information see:
        julearn.transformers.dataframe.DataFrameWrapTransformer

    Returns
    -------
    sklearn.pipeline.Pipeline
        A Pipline in which all entries are wrappend in a
        DataFrameWrapTransformer.
    """
    steps_ready_for_pipe = []
    for i_step, step in enumerate(steps):

        name, estimator = step

        if (i_step == len(steps) - 1) and (hasattr(estimator, 'predict')):
            steps_ready_for_pipe.append([name, estimator])
        else:
            if isinstance(estimator, DataFrameWrapTransformer):
                transformer = estimator
            else:
                transformer = DataFrameWrapTransformer(
                    transformer=estimator,
                    apply_to=apply_to)
            steps_ready_for_pipe.append([name, transformer])

    return Pipeline(steps=steps_ready_for_pipe)


class ExtendedDataFramePipeline(BaseEstimator):
    """A class creating a custom metamodel like a Pipeline. In practice this
    should be created using :ref:julearn.pipeline._create_extended_pipeline.
    There are multiple caveats of creating such a pipline without using
    that function. Compared to an usual scikit-learn pipeline, this have added
    functionalities:

        * Handling transformations of the target:
            The target can be changed. Importantly this transformed target will
            be considered the ground truth to score against.
            Note: if you want to score this pipeline with an external function.
            You have to consider that the scorer needs to be an exteded_scorer.

        * Handling confounds:
            Adds the confound as type to columns.
            This allows the DataFrameWrapTransformer inside of the
            dataframe_pipeline to handle confounds properly.

        * Handling categorical features:
            Adds categorical type to columns so that DataFrameWrapTransformer
            inside of the dataframe_pipline can handle categorical features
            properly.

    column_types are added to the feature dataframe after each column using
    the specified separator.
    E.g. column ``age`` becomes ``age__:type:__confound``.

    Parameters
    ----------
    dataframe_pipeline : obj
        A pipeline working with dataframes and being able to handle confounds.
        Should be created using julearn.pipeline.create_dataframe_pipeline.

    y_transformer : obj or None
        Any transformer which can take the X and y to transform the y.
        You can use julearn.transformers.target.TargetTransfromerWrapper to
        convert most sklearn transformers to a target_transformer.

    confound_dataframe_pipeline : obj or None
        Similar to dataframe_pipeline.
    confounds : list(str) or None
        List of column names which are confounds (defaults to None).
    categorical_features : list(str), optional
        List of column names which are categorical features (defaults to None).

    """

    column_type_sep = '__:type:__'

    def __init__(self, dataframe_pipeline,
                 y_transformer=None,
                 confound_dataframe_pipeline=None,
                 confounds=None, categorical_features=None,
                 ):

        self.dataframe_pipeline = dataframe_pipeline
        self.y_transformer = y_transformer
        self.confound_dataframe_pipeline = confound_dataframe_pipeline
        self.confounds = confounds
        self.categorical_features = categorical_features

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

        if hasattr(self.dataframe_pipeline, 'classes_'):
            self.classes_ = self.dataframe_pipeline.classes_
        return self

    def predict(self, X):
        X_conf_trans = self.transform_confounds(X)
        return self.dataframe_pipeline.predict(X_conf_trans)

    def predict_proba(self, X):
        X_conf_trans = self.transform_confounds(X)
        return self.dataframe_pipeline.predict_proba(X_conf_trans)

    def decision_function(self, X):
        X_conf_trans = self.transform_confounds(X)
        return self.dataframe_pipeline.decision_function(X_conf_trans)

    def score(self, X, y):
        X_conf_trans = self.transform_confounds(X)
        y_true = self.transform_target(X, y)
        return self.dataframe_pipeline.score(X_conf_trans, y_true)

    def transform(self, X):
        X_conf_trans = self.transform_confounds(X)
        X_trans = self.dataframe_pipeline.transform(X_conf_trans)
        return X_trans

    def transform_target(self, X, y):
        X_conf_trans = self.transform_confounds(X)
        y_true = self._transform_target(X_conf_trans, y)
        return y_true

    def transform_confounds(self, X):
        X = self._recode_columns(X)
        if self.confounds is None or self.confound_dataframe_pipeline is None:
            return X
        else:
            return self.confound_dataframe_pipeline.transform(X)

    def preprocess(self, X, y, until=None, return_trans_column_type=False):
        """

        Parameters
        ----------
        until : str or None
            the name of the transformer until which preprocess
            should transform, by default None
        return_trans_column_type : bool or None
            whether to return transformed column names with the associated
            column type, by default False

        Returns
        -------
        tuple(pd.DataFrame, pd.Series)
            Features and target after preprocessing.
        """
        # TODO in case no model at the end
        old_model = self.dataframe_pipeline.steps.pop()
        if until is None:
            X_trans = self.transform(X)
            y_trans = self.transform_target(X, y)
        else:
            try:
                self[until]
            except KeyError:
                raise_error(f'{until} is not a valid step')

            if until.startswith('confound__'):
                step_name = until.replace('confound__', '')
                X_trans = self._transform_pipeline_until(
                    pipeline=self.confound_dataframe_pipeline,
                    step_name=step_name,
                    X=X
                )
                y_trans = y.copy()

            elif until.startswith('target__'):
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
        if not return_trans_column_type:
            X_trans = self._remove_column_types(X_trans)
        return X_trans, y_trans

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **params):
        super().set_params(**{self._rename_param(param): val
                              for param, val in params.items()})
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)

        def _rename_get(param):
            first, *rest = param.split('__')
            # if rest is empty then these are the actual parameters/pipelines
            if rest != []:
                if first == 'dataframe_pipeline':
                    first = []
                elif first == 'confound_dataframe_pipeline':
                    first = ['confounds']
                else:
                    first = ['target']
                return '__'.join(first + rest)
            else:
                return param

        return {_rename_get(param): val
                for param, val in params.items()}

    @ property
    def named_steps(self):
        steps = self.dataframe_pipeline.named_steps
        return Bunch(**
                     {name: self._get_wrapped_step(step)
                         for name, step in steps.items()
                      })

    @ property
    def named_confound_steps(self):
        steps = self.confound_dataframe_pipeline.named_steps
        return Bunch(**
                     {name: self._get_wrapped_step(step)
                         for name, step in steps.items()
                      })

    def __getitem__(self, ind):
        if not isinstance(ind, str):
            raise_error('Indexing must be done using strings')
        if ind.startswith('confound__'):
            n_ind = ind.replace('confound__', '')
            element = self.confound_dataframe_pipeline[n_ind]
        elif ind.startswith('target__'):
            element = self.y_transformer
        else:
            element = self.dataframe_pipeline[ind]
        return self._get_wrapped_step(element)

    def __repr__(self):
        preprocess_X = clone(self.dataframe_pipeline).steps
        model = preprocess_X.pop()

        preprocess_X = None if preprocess_X == [] else preprocess_X
        preprocess_target = self.y_transformer
        preprocess_confounds = self.confound_dataframe_pipeline
        categorical_features = (None if self.categorical_features == []
                                else self.categorical_features)

        return f'''
        ExtendedDataFramePipeline using:
            * model = {model}
            * preprocess_X = {preprocess_X}
            * preprocess_target = {preprocess_target}
            * preprocess_confounds = {preprocess_confounds}
            * confounds = {self.confounds}
            * categorical_features = {categorical_features}
              '''

    def _rename_param(self, param):
        first, *rest = param.split('__')
        steps = list(self.named_steps.keys())

        if first in steps:
            new_first = f'dataframe_pipeline__{first}'
        elif first == 'confounds':
            new_first = 'confound_dataframe_pipeline'
        elif first == 'target':
            new_first = 'y_transformer'
        else:
            raise_error(
                'Each element of the hyperparameters dict  has to start with '
                f'"confounds__", "target__" or any of "{steps}__" '
                f'but was {first}')
        return '__'.join([new_first] + rest)

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

    def _recode_columns(self, X):
        return X.rename(columns=self.col_name_mapper_).copy()

    def _transform_pipeline_until(self, pipeline, step_name, X):
        X_transformed = X.copy()
        X_transformed = self._recode_columns(X_transformed)
        for name, step in pipeline.steps:
            X_transformed = step.transform(X_transformed)
            if name == step_name:
                break
        return X_transformed

    def _remove_column_types(self, X_trans):

        inverse_col_name_mapper = {
            col: col.split('__:type:__')[0]
            for col in X_trans.columns
        }
        X_trans = (X_trans
                   .rename(columns=inverse_col_name_mapper)
                   .copy()
                   )

        return X_trans

    @staticmethod
    def _get_wrapped_step(step):
        """returns wrapped transformer if a DataFrameWrapTransformer is
        provided.

        Parameters
        ----------
        step : obj
            A step of a DataFramePipeline

        Returns
        -------
        step
            step if not wrapped else wrapped transformer
        """
        step = (step.transformer
                if isinstance(step, DataFrameWrapTransformer)
                else step
                )
        return step


def _create_extended_pipeline(
    preprocess_steps_features,
    preprocess_transformer_target,
    preprocess_steps_confounds,
    model, confounds, categorical_features
):
    """

    Parameters
    ----------
    preprocess_steps_feature: list(tuple)
        A list of steps. Each step contains a name and  transformer.
        These transformers are applied to the complete feature space or a
        subset of it. For more information look at steps in:
        sklearn.pipeline.Pipeline

    preprocess_transformer_target : y_transform
        A transformer, which takes in X, y and outputs a transformed y.
        Applied after preprocess_steps_confounds, but before
        preprocess_steps_feature

    preprocess_steps_confounds : list(tuple)
        A list of steps. Each step contains a name and  transformer.
        These transformers are applied only to the confounds before
        transforming the target.
    model : tuple(str, obj)
        tuple of name and sklearn estimator
    confounds : list(str) or str
        A list of column_names which are the confounds
        or the column_name of one confound

    categorical_features : list(str) or str
        A list of column_names which are the categorical features
        or the column_name of one categorical feature
    """

    drop_confounds = DataFrameWrapTransformer(
        transformer=DropColumns(columns='.*__:type:__confound'),
        apply_to='all')
    X_steps = (
        (list(preprocess_steps_features) +
         [('drop_confounds', drop_confounds)] +
         [model])
        if preprocess_steps_features is not None
        else [(
            'drop_confounds', drop_confounds),
            model]
    )
    pipeline = create_dataframe_pipeline(X_steps)

    if preprocess_steps_confounds is not None:
        confound_pipe = create_dataframe_pipeline(
            preprocess_steps_confounds,
            apply_to='confound')
    else:
        confound_pipe = None

    return ExtendedDataFramePipeline(
        dataframe_pipeline=pipeline,
        y_transformer=preprocess_transformer_target,
        confound_dataframe_pipeline=confound_pipe,
        confounds=confounds,
        categorical_features=categorical_features)
