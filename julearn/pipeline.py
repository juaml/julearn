# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: BSD

from inspect import getfullargspec
from julearn.utils.array import ensure_2d
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from . transformers.confounds import BaseConfoundRemover
from . utils import raise_error


def make_pipeline(steps, confound_steps=None, y_transformer=None):
    pipeline = Pipeline(steps=steps)
    confound_pipeline = None
    if confound_steps is not None:
        confound_pipeline = Pipeline(confound_steps)
    return ExtendedPipeline(
        pipeline=pipeline, y_transformer=y_transformer,
        confound_pipeline=confound_pipeline)


class ExtendedPipeline(BaseEstimator):
    """A class creating a custom metamodel like a Pipeline. In practice this
    should be created using :ref:julearn.pipeline._create_extended_pipeline.
    There are multiple caveats of creating such a pipline without using
    that function. Compared to an usual scikit-learn pipeline, this have added
    functionalities:

        * Handling transformations of the target:
            The target can be changed. Importantly this transformed target will
            be considered the ground truth to score against.
            Note: if you want to score this pipeline with an external function.
            You have to consider that the scorer needs to be an
            extended_scorer.

        * Handling confounds:
            Adds the confound as type to columns.
            This allows the DataFrameWrapTransformer inside of the
            pipeline to handle confounds properly.

        * Handling categorical features:
            Adds categorical type to columns so that DataFrameWrapTransformer
            inside of the dataframe_pipline can handle categorical features
            properly.

    column_types are added to the feature dataframe after each column using
    the specified separator.
    E.g. column ``age`` becomes ``age__:type:__confound``.

    Parameters
    ----------
    pipeline : obj
        A pipeline working with dataframes and being able to handle confounds.
        Should be created using julearn.pipeline.create_pipeline.

    y_transformer : obj or None
        Any transformer which can take the X and y to transform the y.
        You can use julearn.transformers.target.TargetTransfromerWrapper to
        convert most sklearn transformers to a target_transformer.

    confound_pipeline : obj or None
        Similar to pipeline.

    """

    def __init__(self, pipeline, y_transformer=None, confound_pipeline=None):
        self.pipeline = pipeline
        self.y_transformer = y_transformer
        self.confound_pipeline = confound_pipeline

    def _preprocess(self, X, y=None, **fit_params):
        confounds = fit_params.pop('confounds', None)
        confounds = ensure_2d(confounds)
        all_params = self._split_params(fit_params)
        self.pipeline = clone(self.pipeline)  # TODO check whether needed
        if self.confound_pipeline is not None:
            self.confound_pipeline = clone(self.confound_pipeline)

        if self.y_transformer is not None:
            clone(self.y_transformer)

        fit_params = all_params['pipeline']
        # Preprocess confounds (if specified)
        trans_confounds = confounds
        if self.confound_pipeline is not None:
            trans_confounds = self.confound_pipeline.fit_transform(
                confounds, y, **all_params['confounds'])

        for name, object in self.pipeline.steps:
            if isinstance(object, BaseConfoundRemover):
                fit_params[f'{name}__confounds'] = trans_confounds

        # Tranform y (if specified)
        if self.y_transformer is not None and y is not None:
            y_true = self.y_transformer.fit_transform(
                ensure_2d(y), **all_params['target']).squeeze()
        else:
            y_true = y

        return y_true, fit_params

    def fit(self, X, y=None, **fit_params):
        if fit_params is None:
            fit_params = {}
        y_true, fit_params = self._preprocess(X, y, **fit_params)

        self.pipeline.fit(X, y_true, **fit_params)

        # Copy some of the pipeline attributes
        if hasattr(self.pipeline, 'classes_'):
            self.classes_ = self.pipeline.classes_  # type: ignore

        return self

    def _fit(self, X, y=None, **fit_params):
        y_true, fit_params = self._preprocess(X, y, **fit_params)
        fit_params_steps = self.pipeline._check_fit_params(**fit_params)
        Xt = self.pipeline._fit(X, y_true, **fit_params_steps)

        return Xt

    def _transform(self, X, confounds=None, with_final=True):
        confounds_trans = None
        confounds_trans = self.transform_confounds(confounds)
        X_trans = X
        for _, _, transform in self.pipeline._iter(with_final=with_final):
            if isinstance(transform, BaseConfoundRemover):
                X_trans = transform.transform(
                    X_trans, confounds=confounds_trans)
            else:
                X_trans = transform.transform(X_trans)
        return X_trans

    def transform(self, X, confounds=None):
        """Apply transforms, and transform with the final estimator
        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Adapted from sklearn.Pipeline.transform

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        if self.pipeline._final_estimator != 'passthrough':
            # Check that the attribute exists
            self.pipeline._final_estimator.transform

        return self._transform(X, confounds=confounds, with_final=True)

    def transform_target(self, X, y, confounds=None):
        y_true = ensure_2d(y)
        confounds = ensure_2d(confounds)
        if self.y_transformer is not None:
            args = getfullargspec(self.y_transformer.transform).args
            if 'confounds' in args:
                conf_trans = self.transform_confounds(confounds)
                y_true = self.y_transformer.transform(
                    y_true, confounds=conf_trans)
            else:
                y_true = self.y_transformer.transform(y_true)
        return y_true.squeeze()

    def transform_confounds(self, confounds):
        if confounds is None:
            return confounds
        confounds = ensure_2d(confounds)
        trans_confounds = confounds
        if self.confound_pipeline is not None:
            trans_confounds = self.confound_pipeline.transform(confounds)
        return trans_confounds

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator
        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Adapted from sklearn.Pipeline.fit_transform

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Transformed samples
        """
        if fit_params is None:
            fit_params = {}
        Xt = self._fit(X, y, **fit_params)
        last_step = self.pipeline._final_estimator
        if last_step == "passthrough":
            return Xt
        if 'confounds' in fit_params:
            fit_params.pop('confounds')
        fit_params_steps = self.pipeline._check_fit_params(**fit_params)
        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        if hasattr(last_step, "fit_transform"):
            return last_step.fit_transform(Xt, y, **fit_params_last_step)
        else:
            return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

    def _prepare_predict(self, X, confounds):
        conf_trans = self.transform_confounds(confounds)
        Xt = self._transform(X, confounds=conf_trans, with_final=False)
        return Xt

    def predict(self, X, confounds=None, **predict_params):
        Xt = self._prepare_predict(X, confounds=confounds)
        return self.pipeline.steps[-1][1].predict(Xt, **predict_params)

    def predict_proba(self, X, confounds=None, **predict_params):
        Xt = self._prepare_predict(X, confounds=confounds)
        return self.pipeline.steps[-1][1].predict_proba(Xt, **predict_params)

    def score(self, X, y=None, confounds=None, sample_weight=None):
        conf_trans = self.transform_confounds(confounds=confounds)
        y_true = self.transform_target(X, y)
        Xt = self._transform(
            X, confounds=conf_trans, with_final=False)
        return self.pipeline.steps[-1][1].score(
            Xt, y=y_true, sample_weight=sample_weight)

    def fit_predict(self, X, y=None, **fit_params):
        if fit_params is not None:
            fit_params = {}
        Xt = self._fit(X, y, **fit_params)
        fit_params_steps = self.pipeline._check_fit_params(**fit_params)
        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        return self.pipeline.steps[-1][1].fit_predict(
            Xt, y, **fit_params_last_step)

    def preprocess(self, X, y, confounds=None, until=None):
        """

        Parameters
        ----------
        until : str or None
            the name of the transformer until which preprocess
            should transform, by default None

        Returns
        -------
        X_trans : numpy.ndarray
            The transformed features
        y_trans : numpy.ndarray
            The transformed targets
        conf_trans : numpy.ndarray
            The transformed confounds
        """

        transformers_pipe = clone(self.pipeline)
        if until is None:
            until = transformers_pipe.steps[-2][0]
        try:
            self[until]
        except KeyError:
            raise_error(f'{until} is not a valid step')
        conf_trans = None if confounds is None else ensure_2d(confounds.copy())
        y_trans = y.copy()
        X_trans = X.copy()

        if until.startswith('confound__'):
            if self.confound_pipeline is not None:
                step_name = until.replace('confound__', '')
                for name, step in self.confound_pipeline.steps:
                    conf_trans = step.transform(conf_trans)
                    if name == step_name:
                        break
        elif until.startswith('target__'):
            conf_trans = self.transform_confounds(conf_trans)
            y_trans = self.transform_target(X, y, confounds=conf_trans)
        else:
            conf_trans = self.transform_confounds(conf_trans)
            y_trans = self.transform_target(X, y, confounds=conf_trans)
            for name, step in self.pipeline.steps:
                if isinstance(step, BaseConfoundRemover):
                    X_trans = step.transform(X_trans, confounds=conf_trans)
                else:
                    X_trans = step.transform(X_trans)
                if name == until:
                    break
        return X_trans, y_trans, conf_trans

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
                if first == 'pipeline':
                    first = []
                elif first == 'confound_pipeline':
                    first = ['confounds']
                else:
                    first = ['target']
                return '__'.join(first + rest)
            else:
                return param

        return {_rename_get(param): val
                for param, val in params.items()}

    @property
    def steps(self):
        all_steps = []
        if self.confound_pipeline is not None:
            all_steps.extend(self.confound_pipeline.steps)
        if self.y_transformer is not None:
            all_steps.append(('y_transformer', self.y_transformer))
        all_steps.extend(self.pipeline.steps)
        return all_steps

    @ property
    def named_steps(self):
        return self.pipeline.named_steps

    @ property
    def named_confound_steps(self):
        steps = None
        if self.confound_pipeline is not None:
            steps = self.confound_pipeline.named_steps
        return steps

    def __getitem__(self, ind):
        if not isinstance(ind, str):
            raise_error('Indexing must be done using strings')
        if ind.startswith('confound__'):
            if self.confound_pipeline is None:
                raise_error('No confound pipeline to index')
            n_ind = ind.replace('confound__', '')
            element = self.confound_pipeline[n_ind]
        elif ind.startswith('target__'):
            element = self.y_transformer
        else:
            element = self.pipeline[ind]
        return element

    def __repr__(self):
        preprocess_X = clone(self.pipeline).steps
        model = preprocess_X.pop()
        preprocess_X = None if preprocess_X == [] else preprocess_X
        return f'''
ExtendedPipeline using:
    * model = {model}
    * preprocess_X = {preprocess_X}
    * preprocess_target = {self.y_transformer}
    * preprocess_confounds = {self.confound_pipeline}
'''

    def _rename_param(self, param):
        # Map from "confounds__", "target__" and step to the proper parameter
        first, *rest = param.split('__')
        steps = list(self.named_steps.keys())

        if first in steps:
            new_first = f'pipeline__{first}'
        elif first == 'confounds':
            new_first = 'confound_pipeline'
        elif first == 'target':
            new_first = 'y_transformer'
        else:
            raise_error(
                'Each element of the hyperparameters dict  has to start with '
                f'"confounds__", "target__" or any of "{steps}__" '
                f'but was {first}')
        return '__'.join([new_first] + rest)

    @ staticmethod
    def _split_params(params):
        split_params = {
            'pipeline': {},
            'confounds': {},
            'target': {}
        }
        for t_key, t_value in params.items():
            first, *rest = t_key.split('__')
            if first in ['confounds', 'target']:
                param_key = '__'.join(rest)
                split_params[first][param_key] = t_value
            else:
                split_params['pipeline'][t_key] = t_value
        return split_params

    @staticmethod
    def _transform_pipeline_until(pipeline, step_name, X, confounds):
        X_transformed = X.copy()
        for name, step in pipeline.steps:
            X_transformed = step.transform(X_transformed, confounds)
            if name == step_name:
                break
        return X_transformed
