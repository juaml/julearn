# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: BSD
import inspect
from operator import attrgetter
import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from . transformers.confounds import BaseConfoundRemover, TargetConfoundRemover
from . transformers.target import (BaseTargetTransformer,
                                   TargetTransformerWrapper)
from . utils import raise_error, warn
from . utils.array import safe_select


def make_pipeline(steps, confound_steps=None, y_transformer=None):
    pipeline = Pipeline(steps=steps)
    confound_pipeline = None
    if confound_steps is not None:
        confound_pipeline = Pipeline(confound_steps)
    if y_transformer is not None:
        if not isinstance(y_transformer, BaseTargetTransformer):
            y_transformer = TargetTransformerWrapper(y_transformer)
    return ExtendedPipeline(
        pipeline=pipeline, y_transformer=y_transformer,
        confound_pipeline=confound_pipeline)


class ExtendedPipeline(BaseEstimator):
    """A class creating a custom metamodel like a Pipeline.
    There are multiple caveats of creating such a pipline without using
    that function. Compared to an usual scikit-learn pipeline, this have added
    functionalities:

        * Handling transformations of the target:
            The target can be changed. Importantly, the transformed target will
            be considered the ground truth to score against.
            Note: if you want to score this pipeline with an external function.
            You have to consider that the scorer needs to be an
            extended_scorer.

        * Handling confounds:
            Adds the confound as type to columns.
            This allows the pipeline to handle confounds properly.

    Parameters
    ----------
    pipeline : obj
        Pipeline which is applied to the features.
        Should be created using julearn.pipeline.create_pipeline.

    y_transformer : obj or None
        Any transformer which will be applied to the target variable y.

    confound_pipeline : obj or None
        Similar to pipeline.

    """

    def __init__(self, pipeline, y_transformer=None, confound_pipeline=None):
        self.confound_pipeline = confound_pipeline
        self.y_transformer = y_transformer
        self._pipeline = pipeline

        wrapped_steps = []
        for name, transformer in pipeline.steps:
            if (not isinstance(transformer, BaseConfoundRemover) and
                    is_transformable(transformer)):
                wrapped_trans = ColumnTransformer(
                    [(name, transformer, slice(None))],
                    remainder='passthrough')
                wrapped_steps.append((name, wrapped_trans))
            else:
                wrapped_steps.append((name, transformer))
        self.w_pipeline_ = Pipeline(wrapped_steps)

    @property
    def pipeline(self):
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        if calframe[3][3] != 'clone':
            warn(
                "The attribute 'pipeline' should not be accessed. "
                "This parameter will always be the unmodified and unfitted "
                "pipeline")
        return self._pipeline

    def _preprocess(self, X, y=None, **fit_params):
        """ Transform confounds and X, as well as prepare the pipeline for
            selecting the right columns.
        """
        n_confounds = fit_params.pop('n_confounds', 0)
        self.n_confounds_ = n_confounds

        all_params = self._split_params(fit_params)
        # TODO check whether this is needed
        self.w_pipeline_ = clone(self.w_pipeline_)
        if self.confound_pipeline is not None:
            self.confound_pipeline = clone(self.confound_pipeline)

        if self.y_transformer is not None:
            clone(self.y_transformer)

        fit_params = all_params['pipeline']
        # Preprocess confounds (if specified)
        X = self._fit_transform_confounds(X, y, **all_params['confounds'])

        # fit tranform y (if specified)
        y_true = y
        if self.y_transformer is not None and y is not None:
            if isinstance(self.y_transformer, TargetConfoundRemover):
                all_params['target']['n_confounds'] = n_confounds
            y_true = self._fit_transform_target(X, y, **all_params['target'])

        fit_params = self._update_w_pipeline(X, fit_params)

        return X, y_true, fit_params

    def _update_w_pipeline(self, X, fit_params):
        n_features = X.shape[1]
        # Iterate over the pipeline and set the right columns according to
        # the flow of the confounds and features
        n_confounds = self.n_confounds_
        slice_end = n_features - n_confounds if n_confounds > 0 else n_features
        for name, object in self.w_pipeline_.steps:
            if isinstance(object, BaseConfoundRemover):
                # If its a confound remover, will set the number of confounds
                fit_params[f'{name}__n_confounds'] = n_confounds

                # Check if the object will drop confounds
                if object.will_drop_confounds():
                    # If confounds are drop, there will be no more confounds
                    slice_end = n_features
                    # Not really needed, but if more than one confound remover
                    # is to be applied, will fail if the any but the last one
                    # drops the confounds.
                    n_confounds = 0
            elif isinstance(object, ColumnTransformer):
                object.transformers = [
                    (n, t, slice(0, slice_end))
                    for n, t, _ in object.transformers
                ]
        return fit_params

    def fit(self, X, y=None, **fit_params):
        """Fit the model
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        First the confounds will be fitted and transformed. Then the target
        will be transformed. Finally, the internal pipeline will be
        transformed.

        Adapted from sklearn.Pipeline.fit

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
        Returns
        -------
        self : Pipeline
            This estimator
        """
        if fit_params is None:
            fit_params = {}
        X, y_true, fit_params = self._preprocess(X, y, **fit_params)

        self.w_pipeline_.fit(X, y_true, **fit_params)

        # Copy some of the pipeline attributes
        if hasattr(self.w_pipeline_, 'classes_'):
            self.classes_ = self.w_pipeline_.classes_  # type: ignore

        return self

    def _fit(self, X, y=None, **fit_params):
        X, y_true, fit_params = self._preprocess(X, y, **fit_params)
        fit_params_steps = self.w_pipeline_._check_fit_params(**fit_params)
        Xt = self.w_pipeline_._fit(X, y_true, **fit_params_steps)

        return Xt

    def transform(self, X):
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
        X_trans = self.transform_confounds(X)
        return self.w_pipeline_.transform(X_trans)

    def _fit_transform_target(self, X, y, **target_params):
        y_true = y
        if self.y_transformer is not None:
            y_true = self.y_transformer.fit_transform(X, y, **target_params)
        return y_true

    def transform_target(self, X, y):
        y_true = y
        if self.y_transformer is not None:
            y_true = self.y_transformer.transform(X, y)
        return y_true

    def _fit_transform_confounds(self, X, y, **confound_params):
        if self.n_confounds_ > 0 and self.confound_pipeline is not None:
            # Find the IDX and update the pipelines
            confounds = safe_select(X, slice(-self.n_confounds_, None))
            trans_confounds = self.confound_pipeline.fit_transform(
                confounds, y, **confound_params)
            newX = safe_select(X, slice(None, -self.n_confounds_))
            X = np.c_[newX, trans_confounds]
        return X

    def transform_confounds(self, X):
        if self.n_confounds_ > 0 and self.confound_pipeline is not None:
            # Find the IDX and update the pipelines
            confounds = safe_select(X, slice(-self.n_confounds_, None))
            trans_confounds = self.confound_pipeline.transform(confounds)
            newX = safe_select(X, slice(None, -self.n_confounds_))
            X = np.c_[newX, trans_confounds]
        return X

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
        last_step = self.w_pipeline_._final_estimator
        if last_step == "passthrough":
            return Xt
        if 'n_confounds' in fit_params:
            fit_params.pop('n_confounds')
        fit_params_steps = self.w_pipeline_._check_fit_params(**fit_params)
        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        if hasattr(last_step, "fit_transform"):
            return last_step.fit_transform(Xt, y, **fit_params_last_step)
        else:
            return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

    def predict(self, X, **predict_params):
        X = self.transform_confounds(X)
        return self.w_pipeline_.predict(X, **predict_params)

    def predict_proba(self, X, **predict_params):
        X = self.transform_confounds(X)
        return self.w_pipeline_.predict_proba(X, **predict_params)

    def score(self, X, y=None, sample_weight=None):
        X = self.transform_confounds(X)
        y_true = self.transform_target(X, y)
        return self.w_pipeline_.score(X, y_true, sample_weight=None)

    def fit_predict(self, X, y=None, **fit_params):
        if fit_params is not None:
            fit_params = {}
        Xt = self._fit(X, y, **fit_params)
        fit_params_steps = self.w_pipeline_._check_fit_params(**fit_params)
        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        return self.w_pipeline_.steps[-1][1].fit_predict(
            Xt, y, **fit_params_last_step)

    def preprocess(self, X, y, until=None):
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

        transformers_pipe = clone(self.w_pipeline_)
        if until is None:
            until = transformers_pipe.steps[-2][0]
        try:
            self[until]
        except KeyError:
            raise_error(f'{until} is not a valid step')

        confounds = safe_select(X, slice(-self.n_confounds_, None))
        X_trans = X
        y_trans = y.copy()

        if until.startswith('confound__'):
            if self.confound_pipeline is not None:
                step_name = until.replace('confound__', '')
                for name, step in self.confound_pipeline.steps:
                    confounds = step.transform(confounds)
                    if name == step_name:
                        break
        elif until.startswith('target__'):
            X_trans = self.transform_confounds(X_trans)
            confounds = safe_select(X_trans, slice(-self.n_confounds_, None))
            y_trans = self.transform_target(X_trans, y)
        else:
            X_trans = self.transform_confounds(X_trans)
            confounds = safe_select(X_trans, slice(-self.n_confounds_, None))

            y_trans = self.transform_target(X_trans, y)
            for name, t_step in self.w_pipeline_.steps:
                X_trans = t_step.transform(X_trans)
                if name == until:
                    break
        return X_trans, y_trans, confounds

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
        all_steps.extend(self.w_pipeline_.steps)
        return all_steps

    @ property
    def named_steps(self):
        return self.w_pipeline_.named_steps

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
            element = self.w_pipeline_[ind]
        return element

    def __repr__(self):
        preprocess_X = clone(self.w_pipeline_).steps
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


def is_transformable(t):
    can_transform = hasattr(t, "fit_transform") or hasattr(t, "transform")
    return can_transform
