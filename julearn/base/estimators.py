"""Base classes for julearn estimators."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _check_fit_params

from ..base.column_types import make_type_selector
from ..utils import raise_error
from ..utils.typing import DataLike, ModelLike
from .column_types import ColumnTypes, ColumnTypesLike, ensure_column_types


def _wrapped_model_has(attr):
    """Create a function to check if self.model_ has a given attribute.

    This function is usable by
    :func:`sklearn.utils.metaestimators.available_if`

    Parameters
    ----------
    attr : str
        The attribute to check for.

    Returns
    -------
    check : function
        The check function.

    """

    def check(self):
        """Check if self.model_ has a given attribute.

        Returns
        -------
        bool
            True if self.model_ has the attribute, False otherwise.
        """
        return hasattr(self.model_, attr)

    return check


def _ensure_dataframe(X: DataLike) -> pd.DataFrame:
    """Ensure that the input is a pandas DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        The input to check.

    Returns
    -------
    pd.DataFrame
        The input as a pandas DataFrame.
    """
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


class JuBaseEstimator(BaseEstimator):
    """Base class for julearn estimators.

    Every julearn estimator is aware of the column types of the data. Thus,
    they should be able to provide the column types they need and the column
    types they apply to.

    The main difference between this class and
    :class:`sklearn.base.BaseEstimator` is that this class knows which columns
    to use from the data for its purpose. That is, the `apply_to` and
    `needed_types` attributes.

    Parameters
    ----------
    apply_to : str or list of str or set of str or ColumnTypes
        The column types to apply the estimator to.
    needed_types : str or list of str or set of str or ColumnTypes
        The column types needed by the estimator. If None, there are no
        needed types (default is None)
    """

    def __init__(
        self,
        apply_to: ColumnTypesLike,
        needed_types: Optional[ColumnTypesLike] = None,
    ):
        self.apply_to = apply_to
        self.needed_types = needed_types

    def get_needed_types(self) -> ColumnTypes:
        """Get the column types needed by the estimator.

        Returns
        -------
        ColumnTypes
            The column types needed by the estimator.
        """
        needed_types = self.get_apply_to().copy()
        if self.needed_types is not None:
            needed_types.add(ensure_column_types(self.needed_types))
        return needed_types

    def get_apply_to(self) -> ColumnTypes:
        """Get the column types the estimator applies to.

        Returns
        -------
        ColumnTypes
            The column types the estimator applies to.
        """
        return ensure_column_types(self.apply_to)

    def filter_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get the `apply_to` columns of a pandas DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to filter.

        Returns
        -------
        pd.DataFrame
            The DataFrame with only the `apply_to` columns.
        """
        self._filter = self.get_apply_to().to_type_selector()
        columns = self._filter(X)
        return _ensure_dataframe(X[columns])


class JuTransformer(JuBaseEstimator, TransformerMixin):
    """Base class for julearn transformers.

    Parameters
    ----------
    apply_to : str or list of str or set of str or ColumnTypes
        The column types to apply the estimator to.
    needed_types : str or list of str or set of str or ColumnTypes
        The column types needed by the estimator. If None, there are no
        needed types (default is None)
    row_select_col_type : str or list of str or set of str or ColumnTypes
        The column types needed to select rows (default is None)
    row_select_vals : str, int, bool or list of str, int, bool
        The value(s) which should be selected in the row_select_col_type
        to select the rows used for training (default is None)
    """

    def __init__(
        self,
        apply_to: ColumnTypesLike,
        needed_types: Optional[ColumnTypesLike] = None,
        row_select_col_type:  Optional[ColumnTypesLike] = None,
        row_select_vals:  Optional[Union[str, int, list, bool]] = None,
    ):
        self.apply_to = apply_to
        self.needed_types = needed_types
        self.row_select_col_type = row_select_col_type
        self.row_select_vals = row_select_vals

    def fit(self, X, y=None, **fit_params):

        if self.row_select_col_type is None:
            return self._fit(X, y, **fit_params)

        self._col_to_select_rows = make_type_selector(
            ColumnTypes(self.row_select_col_type)._to_pattern())(X)
        if len(self._col_to_select_rows) != 1:
            raise_error(
                "Only, one column can be selected for row_select_col_type."
            )
        self._col_to_select_rows = self._col_to_select_rows[0]

        if not isinstance(self.row_select_vals, list):
            self.row_select_vals = [self.row_select_vals]
        return self._fit(**self._select_rows(X, y, **fit_params))

    def _add_backed_filtered(
        self, X: pd.DataFrame, X_trans: pd.DataFrame
    ) -> pd.DataFrame:
        """Add the left-out columns back to the transformed data.

        Parameters
        ----------
        X : pd.DataFrame
            The original data.
        X_trans : pd.DataFrame
            The transformed data.

        Returns
        -------
        pd.DataFrame
            The transformed data with the left-out columns added back.

        """
        filtered_columns = self._filter(X)
        non_filtered_columns = [
            col for col in list(X.columns) if col not in filtered_columns
        ]
        return pd.concat((X.loc[:, non_filtered_columns], X_trans), axis=1)

    def _select_rows(self, X, y, **fit_params):
        idx = X.query(
            f"`{self._col_to_select_rows}` in @self.row_select_vals"
        ).index.values
        _X = X.loc[idx, :]
        _y = y if y is None else y.loc[idx]
        fit_params = _check_fit_params(X, fit_params, indices=idx)

        return dict(X=_X, y=_y, **fit_params)

    def get_needed_types(self) -> ColumnTypes:
        """Get the column types needed by the estimator.

        Returns
        -------
        ColumnTypes
            The column types needed by the estimator.
        """
        needed_types = super().get_needed_types()
        if self.row_select_col_type is not None:
            needed_types = needed_types.add(self.row_select_col_type)
        return needed_types


class WrapModel(JuBaseEstimator):
    """Wrap a model to make it a julearn estimator.

    Parameters
    ----------
    model : ModelLike
        The model to wrap.
    apply_to : str or list of str or set of str or ColumnTypes
        The column types to apply the model to. If None, the model is
        applied to `continuous` type (default is None).
    needed_types : str or list of str or set of str or ColumnTypes
        The column types needed by the model. If None, there are no
        needed types (default is None)
    **params
        The parameters to set on the model.
    """

    def __init__(
        self,
        model: ModelLike,
        apply_to: Optional[ColumnTypesLike] = None,
        needed_types: Optional[ColumnTypesLike] = None,
        **params,
    ):
        self.model = model
        if apply_to is None:
            apply_to = "continuous"
        # self.apply_to = apply_to
        # self.needed_types = needed_types
        self.model.set_params(**params)
        super().__init__(apply_to=apply_to, needed_types=needed_types)

    def fit(
        self, X: pd.DataFrame, y: Optional[DataLike] = None, **fit_params: Any
    ) -> "WrapModel":
        """Fit the model.

        This method will fit the model using only the columns selected by
        `apply_to`.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit the model on.
        y : DataLike, optional
            The target data (default is None).
        **fit_params : Any
            Additional parameters to pass to the model's fit method.

        Returns
        -------
        WrapModel
            The fitted model.
        """
        self.apply_to = ensure_column_types(self.apply_to)
        if self.needed_types is not None:
            self.needed_types = ensure_column_types(self.needed_types)

        Xt = self.filter_columns(X)
        self.model_ = self.model
        self.model_.fit(Xt, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> DataLike:
        """Predict using the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.

        Returns
        -------
        DataLike
            The predictions.
        """
        Xt = self.filter_columns(X)
        return self.model_.predict(Xt)

    def score(self, X: pd.DataFrame, y: DataLike) -> float:
        """Score the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.
        y : DataLike
            The true target values.

        Returns
        -------
        float
            The score.
        """
        Xt = self.filter_columns(X)
        return self.model_.score(Xt, y)

    @available_if(_wrapped_model_has("predict_proba"))
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.

        Returns
        -------
        np.ndarray
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        Xt = self.filter_columns(X)
        return self.model_.predict_proba(Xt)  # type: ignore

    @available_if(_wrapped_model_has("decision_function"))
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Evaluate the decision function for the samples in X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to obtain the decision function.

        Returns
        -------
        X : array-like of shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        Xt = self.filter_columns(X)
        return self.model_.decision_function(Xt)  # type: ignore

    @available_if(_wrapped_model_has("predict_log_proba"))
    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.

        Returns
        -------
        np.ndarray
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        Xt = self.filter_columns(X)
        return self.model_.predict_log_proba(Xt)  # type: ignore

    @property
    def classes_(self) -> np.ndarray:
        """Get the classes of the model."""
        return self.model_.classes_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get the parameters of the model.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this model and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep=False)
        model_params = self.model.get_params(deep)
        params.update(model_params)
        return params

    def set_params(self, **kwargs: Any) -> "WrapModel":
        """Set the parameters of this model.

        The method works on simple models as well as on nested objects
        (such as :class:`sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Model parameters.

        Returns
        -------
        WrapModel
            WrapModel instance.
        """
        model_params = list(self.model.get_params(True).keys())
        kwargs = cast(Dict[str, Any], kwargs)
        for param, val in kwargs.items():
            if param in model_params:
                self.model.set_params(**{param: val})
            else:
                setattr(self, param, val)
        return self

    @property
    def _estimator_type(self):
        return getattr(self.model, "_estimator_type", None)
