"""Provide functions for corrected t-test."""

# Author: Authors of scikit-learn
#         Martina G. Vilas <https://github.com/martinagvilas>
#         Federico Raimondo <f.raimondo@fz-juelich.de>
# License: BSD 3 clause

from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.special as special
from statsmodels.stats.multitest import multipletests

from ..utils.checks import check_scores_df
from ..utils.logging import raise_error, warn_with_log


def _corrected_std(
    differences: np.ndarray, n_train: int, n_test: int
) -> float:
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1, axis=0) * (
        1 / kr + n_test / n_train
    )
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def _compute_corrected_ttest(
    differences: np.ndarray,
    n_train: int,
    n_test: int,
    df: Optional[int] = None,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Compute paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = differences.mean(axis=0)
    if df is None:
        df = len(differences) - 1
    std = _corrected_std(differences, n_train=n_train, n_test=n_test)
    t_stat = mean / std
    if alternative == "less":
        p_val = special.stdtr(df, t_stat)
    elif alternative == "greater":
        p_val = special.stdtr(df, -t_stat)
    elif alternative == "two-sided":
        p_val = special.stdtr(df, -np.abs(t_stat)) * 2
    else:
        raise_error(
            f"Invalid alternative {alternative}. Should be "
            "'two-sided', 'less' or 'greater'."
        )
    return t_stat, p_val  # type: ignore


def corrected_ttest(
    *scores: pd.DataFrame,
    df: Optional[int] = None,
    method: str = "bonferroni",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Perform corrected t-test on the scores of two or more models.

    Parameters
    ----------
    *scores : pd.DataFrame
        DataFrames containing the scores of the models. The DataFrames must
        be the output of `run_cross_validation`
    df: int
        Degrees of freedom.
    method : str
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are:

        * `bonferroni` : one-step correction
        * `sidak` : one-step correction
        * `holm-sidak` : step down method using Sidak adjustments
        * `holm` : step-down method using Bonferroni adjustments
        * `simes-hochberg` : step-up method  (independent)
        * `hommel` : closed method based on Simes tests (non-negative)
        * `fdr_bh` : Benjamini/Hochberg  (non-negative)
        * `fdr_by` : Benjamini/Yekutieli (negative)
        * `fdr_tsbh` : two stage fdr correction (non-negative)
        * `fdr_tsbky` : two stage fdr correction (non-negative)

    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.

    """
    scores = check_scores_df(*scores, same_cv=True)
    if len(scores) > 2 and alternative != "two-sided":
        raise_error(
            "Only two-sided tests are supported for more than two models."
        )

    t_scores = [x.set_index(["fold", "repeat"]) for x in scores]

    all_stats = []

    for model_i, model_k in combinations(range(len(t_scores)), 2):
        i_scores = t_scores[model_i]
        k_scores = t_scores[model_k]
        model_i_name = i_scores["model"].iloc[0]
        model_k_name = k_scores["model"].iloc[0]
        n_train = i_scores["n_train"].values
        n_test = i_scores["n_test"].values

        if np.unique(n_train).size > 1:
            warn_with_log(
                "The training set sizes are not the same. Will use a rounded "
                "average."
            )
            n_train = int(np.mean(n_train).round())
        else:
            n_train = n_train[0]

        if np.unique(n_test).size > 1:
            warn_with_log(
                "The testing set sizes are not the same. Will use a rounded "
                "average."
            )
            n_test = int(np.mean(n_test).round())
        else:
            n_test = n_test[0]

        to_skip = ["cv_mdsum", "n_train", "n_test", "model"]

        to_keep = [
            x
            for x in i_scores.columns
            if x not in to_skip
            and (x.startswith("test_") or x.startswith("train_"))
        ]
        df1 = i_scores[to_keep]
        df2 = k_scores[to_keep]
        differences = df1 - df2
        t_stat, p_val = _compute_corrected_ttest(
            differences, n_train=n_train, n_test=n_test, df=df
        )
        stat_df = t_stat.to_frame("t-stat")
        stat_df["p-val"] = p_val
        stat_df["model_1"] = model_i_name
        stat_df["model_2"] = model_k_name

        all_stats.append(stat_df)

    all_stats_df = pd.concat(all_stats)
    all_stats_df.index.name = "metric"
    all_stats_df = all_stats_df.reset_index()

    if len(t_scores) > 2:
        corrected_stats = []
        for t_metric in all_stats_df["metric"].unique():
            metric_df = all_stats_df[all_stats_df["metric"] == t_metric].copy()
            corrected = multipletests(metric_df["p-val"], method=method)
            metric_df["p-val-corrected"] = corrected[1]
            corrected_stats.append(metric_df)

        all_stats_df = pd.concat(corrected_stats)
    else:
        all_stats_df["p-val-corrected"] = all_stats_df["p-val"]
    return all_stats_df
