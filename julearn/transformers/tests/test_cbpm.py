"""Provide tests for the CBPM transformer."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import spearmanr
from pandas.testing import assert_frame_equal

from julearn.transformers import CBPM


def test_CBPM_posneg_correlated_features(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame
) -> None:
    """Test the CBPM transformer with posneg correlated features.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    X_pos = ["sepal_length", "petal_length", "petal_width"]
    X_neg = ["sepal_width"]

    trans_X_posneg = CBPM(corr_sign="posneg", agg_method=np.mean
                          ).fit_transform(X_iris, y_iris)
    trans_man_pos = X_iris[X_pos].values.mean(axis=1)
    trans_man_neg = X_iris[X_neg].values.mean(axis=1)
    trans_man = np.concatenate(
        [trans_man_pos.reshape(-1, 1), trans_man_neg.reshape(-1, 1)], axis=1
    )
    assert_array_equal(trans_X_posneg, trans_man)


def test_CBPM_pos_correlated_features(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame
) -> None:
    """Test the CBPM transformer with positive correlated features.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    X_pos = ["sepal_length", "petal_length", "petal_width"]

    trans_X_pos = CBPM(corr_sign="pos", agg_method=np.mean
                       ).fit_transform(X_iris[X_pos], y_iris)

    trans_X_pos_neg = CBPM(corr_sign="pos", agg_method=np.mean
                           ).fit_transform(X_iris, y_iris)

    trans_man = X_iris[X_pos].values.mean(axis=1)

    assert_array_equal(trans_X_pos, trans_X_pos_neg)
    assert_array_equal(trans_X_pos, trans_man)


def test_CBPM_neg_correlated_features(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame
) -> None:
    """Test the CBPM transformer with positive correlated features.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    X_neg = ["sepal_width"]

    trans_X_neg = CBPM(corr_sign="neg", agg_method=np.mean
                       ).fit_transform(X_iris[X_neg], y_iris)

    trans_X_neg_neg = CBPM(corr_sign="neg", agg_method=np.mean
                           ).fit_transform(X_iris, y_iris)

    trans_man = X_iris[X_neg].values.mean(axis=1)

    assert_array_equal(trans_X_neg, trans_X_neg_neg)
    assert_array_equal(trans_X_neg, trans_man)


def test_CBPM_warnings(X_iris: pd.DataFrame, y_iris: pd.DataFrame) -> None:
    """Test the CBPM transformer warnings.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """
    X_pos = ["sepal_length", "petal_length", "petal_width"]
    X_neg = ["sepal_width"]

    # No negative features, replace with mean
    with pytest.warns(
        RuntimeWarning, match="No feature with significant positive"
    ):
        trans = CBPM(corr_sign="pos", agg_method=np.mean
                     ).fit_transform(X_iris[X_neg], y_iris)

    assert (trans == y_iris.values.mean()).all()

    # No positive features, replace with mean
    with pytest.warns(
        RuntimeWarning, match="No feature with significant negative"
    ):
        trans = CBPM(corr_sign="neg", agg_method=np.mean
                     ).fit_transform(X_iris[X_pos], y_iris)

    assert (trans == y_iris.values.mean()).all()

    # Use posneg, but only positive present
    trans_pos = CBPM(corr_sign="pos", agg_method=np.mean
                     ).fit_transform(X_iris[X_pos], y_iris)
    with pytest.warns(
        RuntimeWarning, match="Only features with positive correlations"
    ):
        trans = CBPM(corr_sign="posneg", agg_method=np.mean
                     ).fit_transform(X_iris[X_pos], y_iris)

    assert_array_equal(trans, trans_pos)

    # Use posneg, but only negative present
    trans_neg = CBPM(corr_sign="neg", agg_method=np.mean
                     ).fit_transform(X_iris[X_neg], y_iris)
    with pytest.warns(
        RuntimeWarning, match="Only features with negative correlations"
    ):
        trans = CBPM(corr_sign="posneg", agg_method=np.mean
                     ).fit_transform(X_iris[X_neg], y_iris)

    assert_array_equal(trans, trans_neg)

    # No features, replace with mean
    df_shuffled_X = X_iris.sample(frac=1, random_state=42)
    with pytest.warns(
        RuntimeWarning,
        match="No feature with significant negative or positive",
    ):
        trans = CBPM(corr_sign="posneg", agg_method=np.mean
                     ).fit_transform(df_shuffled_X, y_iris)
    assert (trans == y_iris.values.mean()).all()


def test_CBPM_lower_sign_threshhold(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame
) -> None:
    """Test the CBPM transformer with lower significance threshold.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """
    trans_posneg = CBPM(
        corr_sign="pos", significance_threshold=1e-50, agg_method=np.mean
    ).fit_transform(X_iris, y_iris)

    # I have checked before that only these 2 have pvalues under 1e-50
    trans_man = X_iris[["petal_length", "petal_width"]].values.mean(axis=1)

    assert_array_equal(trans_posneg, trans_man)


def test_CBPM_lower_sign_threshhold_no_sig(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame
) -> None:
    """Test the CBPM transformer with an even lower significance threshold.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    # I have checked before there are no under 1e-100
    with pytest.warns(
        RuntimeWarning,
        match="No feature with significant negative or positive",
    ):
        trans_posneg = CBPM(
            corr_sign="posneg", significance_threshold=1e-100,
            agg_method=np.mean
        ).fit_transform(X_iris, y_iris)
    assert (trans_posneg == y_iris.values.mean()).all()


def test_CBPM_spearman(X_iris: pd.DataFrame, y_iris: pd.DataFrame) -> None:
    """Test the CBPM transformer with spearman correlation.

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    X_pos = ["sepal_length", "petal_length", "petal_width"]
    X_neg = ["sepal_width"]

    # I have checked before all are still significant with spearman
    trans_posneg = CBPM(corr_method=spearmanr,
                        agg_method=np.mean
                        ).fit_transform(X_iris, y_iris)

    trans_man_pos = X_iris[X_pos].values.mean(axis=1)
    trans_man_neg = X_iris[X_neg].values.mean(axis=1)
    trans_man = np.concatenate(
        [trans_man_pos.reshape(-1, 1), trans_man_neg.reshape(-1, 1)], axis=1
    )
    assert_array_equal(trans_posneg, trans_man)


def test_CBPM_set_output_posneg(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame,
) -> None:
    """

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    X_pos = ["sepal_length", "petal_length", "petal_width"]
    X_neg = ["sepal_width"]

    # I have checked before all are still significant with spearman
    trans_posneg = (CBPM(corr_method=spearmanr,
                         agg_method=np.mean,
                         corr_sign="posneg"
                         )
                    .set_output(transform="pandas")
                    .fit_transform(X_iris, y_iris)
                    )

    trans_man_pos = X_iris[X_pos].values.mean(axis=1)
    trans_man_neg = X_iris[X_neg].values.mean(axis=1)
    trans_man = np.concatenate(
        [trans_man_pos.reshape(-1, 1), trans_man_neg.reshape(-1, 1)], axis=1
    )
    df_trans_man = pd.DataFrame(trans_man, columns=["positive", "negative"])
    assert_frame_equal(trans_posneg, df_trans_man)


def test_CBPM_set_output_pos(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame,
) -> None:
    """

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    X_pos = ["sepal_length", "petal_length", "petal_width"]

    # I have checked before all are still significant with spearman
    trans_pos = (CBPM(corr_method=spearmanr,
                      agg_method=np.mean,
                      corr_sign="pos"
                      )
                 .set_output(transform="pandas")
                 .fit_transform(X_iris, y_iris)
                 )

    trans_man_pos = X_iris[X_pos].values.mean(axis=1)
    df_trans_man = pd.DataFrame(trans_man_pos, columns=["positive"])
    assert_frame_equal(trans_pos, df_trans_man)


def test_CBPM_set_output_neg(
    X_iris: pd.DataFrame, y_iris: pd.DataFrame,
) -> None:
    """

    Parameters
    ----------
    X_iris : pd.DataFrame
        The iris dataset features
    y_iris : pd.Series
        The iris dataset target
    """

    X_neg = ["sepal_width"]

    # I have checked before all are still significant with spearman
    trans_neg = (CBPM(corr_method=spearmanr,
                      agg_method=np.mean,
                      corr_sign="neg"
                      )
                 .set_output(transform="pandas")
                 .fit_transform(X_iris, y_iris)
                 )

    trans_man_neg = X_iris[X_neg].values.mean(axis=1)
    df_trans_man = pd.DataFrame(trans_man_neg, columns=["negative"])
    assert_frame_equal(trans_neg, df_trans_man)
