"""Provide julearn specific column transformer."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from ..base import ColumnTypesLike, JuTransformer, ensure_column_types
from ..utils.logging import raise_error
from ..utils.typing import DataLike, EstimatorLike


class JuColumnTransformer(JuTransformer):
    """Column transformer that can be used in a julearn pipeline.

    This column transformer is a wrapper around the sklearn column transformer,
    so it can be used directly with julearn pipelines.

    Parameters
    ----------
    name : str
        Name of the transformer.
    transformer : EstimatorLike
        The transformer to apply to the columns.
    apply_to : ColumnTypesLike
        To which column types the transformer needs to be applied to.
    needed_types : ColumnTypesLike, optional
        Which feature types are needed for the transformer to work.
    row_select_col_type : str or list of str or set of str or ColumnTypes
        The column types needed to select rows (default is None).
    row_select_vals : str, int, bool or list of str, int, bool
        The value(s) which should be selected in the row_select_col_type
        to select the rows used for training (default is None).
    **params : dict
        Extra keyword arguments for the transformer.

    """

    def __init__(
        self,
        name: str,
        transformer: EstimatorLike,
        apply_to: ColumnTypesLike,
        needed_types: Optional[ColumnTypesLike] = None,
        row_select_col_type: Optional[ColumnTypesLike] = None,
        row_select_vals: Optional[Union[str, int, List, bool]] = None,
        **params: Any,
    ):
        self.name = name
        self.transformer = transformer
        self.apply_to = ensure_column_types(apply_to)
        self.needed_types = needed_types
        self.row_select_col_type = row_select_col_type
        self.row_select_vals = row_select_vals
        self.set_params(**params)

    def _fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Optional[DataLike] = None,
        **fit_params: Any,
    ) -> "JuColumnTransformer":
        """Fit the transformer.

        Fit the transformer to the data, only for the specified columns.

        Parameters
        ----------
        X : np.array
            Input features.
        y : np.array
            Target.
        **fit_params : dict
            Parameters for fitting the transformer.

        Returns
        -------
        JuColumnTransformer
            The fitted transformer.

        """
        verbose_feature_names_out = isinstance(
            self.transformer, ClassNamePrefixFeaturesOutMixin
        )

        self.column_transformer_ = ColumnTransformer(
            [(self.name, self.transformer, self.apply_to.to_type_selector())],
            verbose_feature_names_out=verbose_feature_names_out,
            remainder="passthrough",
        )
        self.column_transformer_.fit(X, y, **fit_params)

        return self

    def transform(self, X: pd.DataFrame) -> DataLike:  # noqa: N803
        """Apply the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed.

        Returns
        -------
        pd.DataFrame
            Transformed data.

        """
        check_is_fitted(self)
        return self.column_transformer_.transform(X)  # type: ignore

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get names of features to be returned.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features to use.

            * If ``None``, then ``feature_names_in_`` is
              used as input feature names if it's defined. If
              ``feature_names_in_`` is undefined, then the following input
              feature names are generated:
              ``["x0", "x1", ..., "x(n_features_in_ - 1)"]``.
            * If ``array-like``, then ``input_features`` must
              match ``feature_names_in_`` if it's defined.

        Returns
        -------
        list of str
            Names of features to be kept in the output pd.DataFrame.

        """
        out = None
        try:
            out = self.column_transformer_.get_feature_names_out(
                input_features
            )
        except ValueError as e:
            raise_error(
                "This transformer changes the names of the features. "
                "Unfortunately, this feature is already present and will "
                "create a repeated feature name. Please re-implement your "
                "transformer, inheriting from "
                "sklearn.base.ClassNamePrefixFeaturesOutMixin",
                klass=ValueError,
                exception=e,
            )
        if self.column_transformer_.verbose_feature_names_out:
            out = [
                x.replace("remainder__", "") if "remainder__" in x else x
                for x in out
            ]
        return out  # type: ignore

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Not used. Kept for compatibility with scikit-learn.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        return dict(
            **self.transformer.get_params(True),
            name=self.name,
            apply_to=self.apply_to,
            needed_types=self.needed_types,
            row_select_col_type=self.row_select_col_type,
            row_select_vals=self.row_select_vals,
            transformer=self.transformer,
        )

    def set_params(self, **kwargs: Any) -> "JuColumnTransformer":
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **kwargs : dict
            Estimator parameters.

        Returns
        -------
        JuColumnTransformer
            JuColumnTransformer instance with params set.

        """
        transformer_params = list(self.transformer.get_params(True).keys())

        for param, val in kwargs.items():
            if param in transformer_params:
                self.transformer.set_params(**{param: val})
            else:
                setattr(self, param, val)
        return self
