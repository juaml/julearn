# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: BSD
from julearn.transformers.available_transformers import (
    _propagate_transformer_column_names)
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.compose import ColumnTransformer
from sklearn.utils.metaestimators import if_delegate_has_method

from . transformers.confounds import BaseConfoundRemover, TargetConfoundRemover
from . transformers.target import (BaseTargetTransformer,
                                   TargetTransformerWrapper)
from . utils import raise_error
from . utils.array import safe_select
from . utils.validation import is_transformable, check_n_confounds


def make_pipeline(steps, confound_steps=None, y_transformer=None):
    if y_transformer is not None:
        if not isinstance(y_transformer, BaseTargetTransformer):
            y_transformer = TargetTransformerWrapper(y_transformer)
    return ExtendedPipeline(
        pipeline_steps=steps, y_transformer=y_transformer,
        confound_pipeline_steps=confound_steps)


class ExtendedPipeline(_BaseComposition):
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

    def __init__(self, pipeline_steps, y_transformer=None,
                 confound_pipeline_steps=None):
        self.pipeline_steps = pipeline_steps
        self.confound_pipeline_steps = confound_pipeline_steps
        self.y_transformer = y_transformer

    def _preprocess(self, X, y=None, **fit_params):
        """ Transform confounds and X, as well as prepare the pipeline for
            selecting the right columns.
        """
        wrapped_steps = []
        # TODO validate pipeline_steps not including _internally_wrapped_
        for name, transformer in self.pipeline_steps:
            if (not isinstance(transformer, BaseConfoundRemover) and
                    is_transformable(transformer)):
                wrapped_trans = ColumnTransformer(
                    [(name, transformer, slice(None))],
                    remainder='passthrough')
                wrapped_steps.append(
                    ('_internally_wrapped_' + name, wrapped_trans))
            else:
                wrapped_steps.append((name, transformer))

        self._pipeline = Pipeline(wrapped_steps)
        self._confound_pipeline = (None if self.confound_pipeline_steps is None
                                   else Pipeline(self.confound_pipeline_steps))
        n_confounds = fit_params.pop('n_confounds', 0)
        check_n_confounds(n_confounds)
        self.n_confounds_ = n_confounds

        all_params = self._split_params(fit_params)
        if self.y_transformer is not None:
            clone(self.y_transformer)

        fit_params = all_params['pipeline']
        # Preprocess confounds (if specified)
        X = self._fit_transform_confounds(X, y, **all_params['confounds'])

        # fit transform y (if specified)
        y_true = y
        if self.y_transformer is not None and y is not None:
            if isinstance(self.y_transformer, TargetConfoundRemover):
                all_params['target']['n_confounds'] = n_confounds

            y_true = self.y_transformer.fit_transform(
                X, y, **all_params['target'])

        fit_params = self._update_w_pipeline(X, fit_params)

        return X, y_true, fit_params

    def _update_w_pipeline(self, X, fit_params):
        n_features = X.shape[1]
        n_confounds = self.n_confounds_
        check_n_confounds(n_confounds)
        slice_end = n_features - n_confounds

        # Iterate over the pipeline and
        # set the columns used by each step, according to
        # the flow of the confounds and features
        for name, object in self._pipeline.steps:
            if isinstance(object, BaseConfoundRemover):
                # If its a confound remover, will set the number of confounds
                fit_params[f'{name}__n_confounds'] = n_confounds

                # Check if the object will drop confounds
                if object.will_drop_confounds():
                    # If confounds are drop, there will be no more confounds
                    slice_end = n_features
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
        X, y_true, fit_params = self._preprocess(X, y, **fit_params)
        self._pipeline.fit(X, y_true, **fit_params)
        self._update_pipeline_steps_from_wrapped()
        # Copy some of the pipeline attributes
        if hasattr(self._pipeline, 'classes_'):
            self.classes_ = self._pipeline.classes_  # type: ignore

        return self

    def _fit(self, X, y=None, **fit_params):
        X, y_true, fit_params = self._preprocess(X, y, **fit_params)
        fit_params_steps = self._pipeline._check_fit_params(**fit_params)
        Xt = self._pipeline._fit(X, y_true, **fit_params_steps)
        self._update_pipeline_steps_from_wrapped()
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
        X_trans = self._pipeline.transform(X_trans)
        X_trans = self._last_trans_drops_confounds(X_trans)
        return X_trans

    def transform_target(self, X, y):
        y_true = y
        if self.y_transformer is not None:
            y_true = self.y_transformer.transform(X, y)
        return y_true

    def _fit_transform_confounds(self, X, y, **confound_params):
        if self.n_confounds_ > 0 and self._confound_pipeline is not None:
            # Find the IDX and update the pipelines
            confounds = safe_select(X, slice(-self.n_confounds_, None))
            trans_confounds = self._confound_pipeline.fit_transform(
                confounds, y, **confound_params)

            # update confound_pipeline_steps
            self.confound_pipeline_steps = self._confound_pipeline.steps
            newX = safe_select(X, slice(None, -self.n_confounds_))
            X = np.c_[newX, trans_confounds]
        return X

    def transform_confounds(self, X):
        if self.n_confounds_ > 0 and self._confound_pipeline is not None:
            # Find the IDX and update the pipelines
            confounds = safe_select(X, slice(-self.n_confounds_, None))
            trans_confounds = self._confound_pipeline.transform(confounds)
            newX = safe_select(X, slice(None, -self.n_confounds_))
            X = np.c_[newX, trans_confounds]
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.
        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first
            step of the pipeline.
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
        Xt = self._fit(X, y, **fit_params)
        last_step = self._pipeline._final_estimator
        if last_step == "passthrough":
            return Xt
        if 'n_confounds' in fit_params:
            fit_params.pop('n_confounds')
        fit_params_steps = self._pipeline._check_fit_params(**fit_params)
        fit_params_last_step = fit_params_steps[self._pipeline.steps[-1][0]]

        if isinstance(last_step, BaseConfoundRemover):
            fit_params_last_step['n_confounds'] = self.n_confounds_

        if hasattr(last_step, "fit_transform"):
            Xt = last_step.fit_transform(Xt, y, **fit_params_last_step)
        else:
            Xt = last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

        if (isinstance(last_step, BaseConfoundRemover) and
                (not last_step.will_drop_confounds())):
            Xt = safe_select(Xt, slice(None, -self.n_confounds_))

        self._update_pipeline_steps_from_wrapped()

        return Xt

    @ if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        X = self.transform_confounds(X)
        return self._pipeline.predict(X, **predict_params)

    @ if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        X = self.transform_confounds(X)
        return self._pipeline.predict_proba(X)

    @ if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        X = self.transform_confounds(X)
        return self._pipeline.decision_function(X)

    def score(self, X, y=None, sample_weight=None):
        X = self.transform_confounds(X)
        y_true = self.transform_target(X, y)
        return self._pipeline.score(X, y_true, sample_weight=None)

    def fit_predict(self, X, y=None, **fit_params):
        Xt = self._fit(X, y, **fit_params)

        Xt = self._last_trans_drops_confounds(Xt)
        # self._fit dealt with n_confound so we remove it
        _ = fit_params.pop('n_confounds')
        fit_params_steps = self._pipeline._check_fit_params(**fit_params)
        fit_params_last_step = fit_params_steps[self._pipeline.steps[-1][0]]
        pred = self._pipeline.steps[-1][1].fit_predict(
            Xt, y, **fit_params_last_step)
        self._update_pipeline_steps_from_wrapped()
        return pred

    def preprocess(self, X, y, until=None, column_names=None):
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

        if until is None:
            until = self.pipeline_steps[-2][0]
        try:
            self[until]
        except KeyError:
            raise_error(f'{until} is not a valid step')
        if column_names is None and isinstance(X, pd.DataFrame):
            column_names = np.array(X.columns)

        confounds = safe_select(X, slice(-self.n_confounds_, None))
        c_column_names = (column_names if column_names is None
                          else column_names[slice(-self.n_confounds_, None)]
                          )
        X_trans = X
        y_trans = y.copy()

        if until.startswith('confounds__'):
            if self._confound_pipeline is not None:
                step_name = until.replace('confounds__', '')
                for name, step in self._confound_pipeline.steps:
                    c_column_names = _propagate_transformer_column_names(
                        step, confounds, c_column_names)
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
            wrapped_until = self._get_wrapped_param_name(until)
            # get wrapper itself
            wrapped_until = wrapped_until.split('__')[0]
            for name, t_step in self._pipeline.steps:

                if column_names is not None:
                    column_names = _propagate_transformer_column_names(
                        t_step, X_trans, column_names)

                X_trans = t_step.transform(X_trans)

                if name == wrapped_until:
                    break
        if column_names is not None:
            X_trans = pd.DataFrame(X_trans, columns=column_names)
        if c_column_names is not None:
            confounds = pd.DataFrame(confounds, columns=c_column_names)
        return X_trans, y_trans, confounds

    def set_params(self, **params):
        for param, val in params.items():

            if param.startswith('confounds__'):
                param_name = param.replace('confounds__', '')
                # set inside of the created pipeline
                if ((hasattr(self, '_confound_pipeline')) and
                        (self._confound_pipeline is None)):
                    raise_error(
                        ('Your confounding pipeline seems to be None '
                         'So you cannot set any parameter for it.'),
                        AttributeError)
                else:
                    try:
                        self._set_params('confound_pipeline_steps',
                                         ** {param_name: val})

                        if hasattr(self, '_confound_pipeline'):
                            self._confound_pipeline.set_params(
                                **{param_name: val})
                    except ValueError:
                        if self.confound_pipeline_steps is None:
                            raise_error(
                                'You cannot set parameters for the confound '
                                'Pipeline as there is None.'
                            )
                        else:
                            possible_params = [
                                'confounds__' + name for name, _ in
                                self.confound_pipeline_steps]  # type: ignore
                            raise_error(
                                f'You cannot set {param} as it is not part of'
                                ' the confounding pipeline. Possible params '
                                f'are: {possible_params}'
                            )

            elif param.startswith('target__'):
                param_name = param.replace('target__', '')
                if self.y_transformer is None:
                    raise_error(
                        ('Your y_transformer seems to be None '
                         'So you cannot set any parameter for it.'),
                        AttributeError)
                else:
                    possible_params = self.y_transformer.get_params().keys()
                    try:
                        self.y_transformer.set_params(**{param_name: val})
                    except ValueError:
                        raise_error(
                            f'You cannot set {param} as it is not a valid '
                            'param of the target transformer.'
                            f'Possible params are: {possible_params}'
                        )

                        # parameters of constructor
            elif param in ['pipeline_steps', 'y_transformer',
                           'confound_pipeline_steps']:
                super().set_params(**{param: val})

            # parameters of pipeline_steps and _pipeline
            else:
                try:
                    self._set_params('pipeline_steps', **{param: val})
                    wrapped_param = self._get_wrapped_param_name(param)

                    if hasattr(self, '_pipeline'):
                        self._pipeline.set_params(**{wrapped_param: val})
                except ValueError:
                    raise_error(
                        f'You cannot set {param} as it is not part of the '
                        'pipeline. Possible params are: '
                        f'{self.get_params()}'
                    )

        return self

    def get_params(self, deep=True):
        init_arguments = ['pipeline_steps',
                          'confound_pipeline_steps',
                          'y_transformer']
        pipeline_steps_params = self._get_params('pipeline_steps', deep=deep)
        confound_pipeline_steps_params = (
            {} if self.confound_pipeline_steps is None
            else self._get_params('confound_pipeline_steps', deep=deep))
        y_transformer_params = (
            {} if self.y_transformer is None
            else self.y_transformer.get_params(deep=deep))

        pipeline_steps_params = {
            param: val
            for param, val in pipeline_steps_params.items()
            if param not in init_arguments
        }
        confound_pipeline_steps_params = {
            (param if param == 'confound_pipeline_steps'
             else 'confounds__' + param): val
            for param, val in confound_pipeline_steps_params.items()
            if param not in init_arguments
        }
        if deep:
            y_transformer_params = {
                'target__' + param: val
                for param, val in y_transformer_params.items()
            }
        else:
            y_transformer_params = {}

        params = {
            **pipeline_steps_params,
            **confound_pipeline_steps_params,
            **y_transformer_params,
            **{'y_transformer': self.y_transformer,
               'pipeline_steps': self.pipeline_steps,
               'confound_pipeline_steps': self.confound_pipeline_steps
               }
        }
        return params

    @ property
    def steps(self):
        all_steps = []
        if self.confound_pipeline_steps is not None:
            all_steps.extend([(f'confounds__{name}', step)
                              for name, step in self.confound_pipeline_steps])
        if self.y_transformer is not None:
            all_steps.append(
                (f'target__{_name_estimators([self.y_transformer])[0][0]}',
                 self.y_transformer)
            )
        all_steps.extend(self.pipeline_steps)
        return all_steps

    @ property
    def named_steps(self):
        if self.confound_pipeline_steps is None:
            conf_dict = {}
        else:
            conf_dict = {f'confounds__{name}': step
                         for name, step in self.confound_pipeline_steps}
        y_dict = (
            {} if self.y_transformer is None
            else {f'target__{_name_estimators([self.y_transformer])[0][0]}':
                  self.y_transformer}
        )

        return Bunch(**dict(self.pipeline_steps),
                     **conf_dict, **y_dict)

    @ property
    def _final_estimator(self):
        estimator = self.steps[-1][1]
        return 'passthrough' if estimator is None else estimator

    def __getitem__(self, ind):
        if not isinstance(ind, str):
            raise_error('Indexing must be done using strings')
        element = self.get_params(deep=True)[ind]
        return element

    def __repr__(self):
        preprocess_X = self.pipeline_steps
        model = preprocess_X.pop()
        preprocess_X = None if preprocess_X == [] else preprocess_X
        return f'''
ExtendedPipeline using:
    * model = {model}
    * preprocess_X = {preprocess_X}
    * preprocess_target = {self.y_transformer}
    * preprocess_confounds = {self.confound_pipeline_steps}
'''

    def _update_pipeline_steps_from_wrapped(self):
        steps = self._pipeline.steps

        # steps_params = {}
        for name, est in steps:

            if name.startswith('_internally_wrapped_'):

                nested_levels = name.split('__')
                if len(nested_levels) > 2:
                    # remove wrapper name
                    steps_name = '__'.join(name.split('__')[1:])
                    self.pipeline_steps = [
                        (inner_name, (est if inner_name == steps_name
                                      else inner_step))
                        for inner_name, inner_step in self.pipeline_steps]

            else:
                self.pipeline_steps = [
                    (inner_name, (est if inner_name == name
                                  else inner_step))
                    for inner_name, inner_step in self.pipeline_steps]

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

    def _get_wrapped_param_name(self, param):

        # find out whether this param was wrapped
        # and adjust param name accordingly
        est_name = param.split('__')[0]
        est = self.get_params()[est_name]
        if (not isinstance(est, BaseConfoundRemover) and
                is_transformable(est)):
            wrapped_param = f'_internally_wrapped_{est_name}__{param}'
        else:
            wrapped_param = param

        return wrapped_param

    def _last_trans_drops_confounds(self, X):
        if ((isinstance(self._final_estimator, BaseConfoundRemover)) or
                ((hasattr(self._final_estimator, 'predict')) and
                 (isinstance(
                     self._pipeline.steps[-2][1], BaseConfoundRemover))
                 )):
            if self.n_confounds_ > 0:
                return safe_select(X, slice(None, -self.n_confounds_))
        return X
