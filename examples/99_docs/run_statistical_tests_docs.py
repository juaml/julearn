# Authors: Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

"""
Statistical tests in machine learning
=====================================

In :ref:`model_comparison_usage` we used ``julearn``'s corrected ``t-test``
to compare models. A question we are often asked is: *why not simply use a
permutation test?* A common practice goes like this: permute the labels
many times, build a "confidence interval" (CI) of the scores of the permuted
(chance) model, and claim that a model is good because its score lies
"outside the CI of the permuted model".

In this section we will see where this reasoning breaks down and which tool is
the right one for each question. In short, there are two *different*
questions:

1. **Is my model better than chance?** This is what a label-permutation test
   answers, as long as it is done correctly.
2. **Is model A better than model B?** This is what the corrected resampled
   ``t-test`` of Nadeau and Bengio [#1]_ answers.

We will go through three parts:

* **Part 1**: the "CI of the permuted model" can be made arbitrarily narrow,
  depending on how it is computed.
* **Part 2**: even when done correctly, label permutation answers a different
  question.
* **Part 3**: when comparing two models, the dependence between CV splits
  must be accounted for, and a permutation test on split-wise scores does not
  do that either.

All the data in this section is synthetic, so we know the ground truth. To
keep the focus on the results, the code that runs the simulations and draws
the figures is not shown, but it is included in the downloadable version of
this page.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    permutation_test_score,
)

from julearn import run_cross_validation
from julearn.stats.corrected_ttest import corrected_ttest
from julearn.utils import configure_logging

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from scipy import stats

# sphinx_gallery_end_ignore

configure_logging(level="ERROR")
X_types = {"continuous": [".*"]}

# sphinx_gallery_start_ignore
# One color per rule/test, used consistently in all the figures
BLUE = "#2a78d6"  # SEM rule / naive t-test
ORANGE = "#eb6834"  # percentile rule / corrected t-test
AQUA = "#1baf7a"  # permutation p-value / sign-flip permutation test
INK = "0.15"  # observed scores
NULL = "0.75"  # null distributions
# sphinx_gallery_end_ignore

###############################################################################
# Part 1: The SEM trap
# --------------------
#
# Let's start with a dataset in which the target ``y`` is *independent* of the
# features ``X``. The two classes are balanced, and no model can do better
# than chance. We evaluate a logistic regression with a 5-fold CV: this is
# the *observed* score. Then, we build the null distribution by permuting the
# labels ``B`` times, repeating the whole CV for each permutation, using
# :func:`sklearn.model_selection.permutation_test_score`.

rng = np.random.default_rng(5)
X_null = [f"f{i}" for i in range(10)]
df_null = pd.DataFrame(rng.normal(size=(100, 10)), columns=X_null)
df_null["y"] = rng.permutation(np.repeat([0, 1], 50))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_null = run_cross_validation(
    X=X_null,
    y="y",
    data=df_null,
    X_types=X_types,
    model="logit",
    problem_type="classification",
    cv=cv,
)
observed = scores_null["test_score"].mean()

_, perm_scores, _ = permutation_test_score(
    LogisticRegression(),
    df_null[X_null],
    df_null["y"],
    cv=cv,
    n_permutations=5000,
    random_state=42,
)

###############################################################################
# Since there is no signal, the observed score is just another draw from the
# null distribution. There are two ways in which a "CI of the permuted model"
# is usually computed:
#
# * **SEM-based**: mean of the null scores :math:`\pm 1.96 \cdot SEM`, with
#   :math:`SEM = SD / \sqrt{B}`.
# * **Percentile-based**: the 2.5th to 97.5th percentile range of the null
#   scores.
#
# Let's draw both on top of the null distribution, using 1000 permutations.

# sphinx_gallery_start_ignore


def plot_null_distribution(t_null, t_observed):
    """Plot the null distribution with the SEM and percentile bands."""
    t_mean = t_null.mean()
    t_sem = t_null.std(ddof=1) / np.sqrt(len(t_null))
    t_low, t_high = np.percentile(t_null, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(8, 4))
    # Accuracies are multiples of 0.01, center the bins on them
    ax.hist(t_null, bins=np.arange(0.295, 0.725, 0.01), color=NULL)
    ax.axvspan(
        t_low,
        t_high,
        color=ORANGE,
        alpha=0.15,
        zorder=0,
        label="2.5-97.5 percentile",
    )
    ax.axvspan(
        t_mean - 1.96 * t_sem,
        t_mean + 1.96 * t_sem,
        color=BLUE,
        alpha=0.7,
        label="mean ± 1.96·SEM",
    )
    ax.axvline(t_observed, color=INK, lw=2, ls="--", label="observed")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of permutations")
    ax.set_title(f"Null distribution, B = {len(t_null)} permutations")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()


# sphinx_gallery_end_ignore
plot_null_distribution(perm_scores[:1000], observed)

###############################################################################
# The percentile band covers the spread of chance-level scores: the observed
# score falls well inside it, as it should. The SEM band is a thin sliver
# around the mean, and the observed score is outside of it. Judging by the SEM
# band, a model *without any signal* is "significant".
#
# The SEM measures how precisely we know the *mean* of the null distribution.
# But the question is whether the observed score is an unusual *draw* from
# that distribution. And the more permutations we run, the more precisely we
# know the mean. Let's see how both bands change with ``B``.

# sphinx_gallery_start_ignore


def plot_bands_vs_permutations(t_perm_scores, t_observed, n_perms):
    """Plot the SEM and percentile bands as a function of B."""
    bands = []
    for n_perm in n_perms:
        t_null = t_perm_scores[:n_perm]
        t_sem = t_null.std(ddof=1) / np.sqrt(n_perm)
        bands.append(
            {
                "sem_low": t_null.mean() - 1.96 * t_sem,
                "sem_high": t_null.mean() + 1.96 * t_sem,
                "pct_low": np.percentile(t_null, 2.5),
                "pct_high": np.percentile(t_null, 97.5),
            }
        )
    bands = pd.DataFrame(bands, index=n_perms)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(
        n_perms,
        bands["pct_low"],
        bands["pct_high"],
        color=ORANGE,
        alpha=0.15,
        label="2.5-97.5 percentile",
    )
    ax.fill_between(
        n_perms,
        bands["sem_low"],
        bands["sem_high"],
        color=BLUE,
        alpha=0.7,
        label="mean ± 1.96·SEM",
    )
    ax.axhline(t_observed, color=INK, lw=2, ls="--", label="observed")
    ax.set_xscale("log")
    ax.set_xlabel("Number of permutations (B)")
    ax.set_ylabel("Accuracy")
    ax.set_title("'CI of the permuted model' vs. number of permutations")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()


# sphinx_gallery_end_ignore
plot_bands_vs_permutations(
    perm_scores, observed, n_perms=[10, 30, 100, 300, 1000, 3000, 5000]
)

###############################################################################
# The SEM band collapses as :math:`1/\sqrt{B}`. With enough permutations, any
# observed score falls outside of it, unless it is *exactly* equal to the
# mean of the null distribution.
#
# The percentile band does not shrink with ``B``. More permutations only make
# its edges more precise. With enough permutations, checking whether the
# observed score lies outside this band is the same as checking whether the
# permutation p-value is below :math:`\alpha = 0.05`. The number of
# permutations does not bias the p-value. It only reduces its Monte Carlo
# noise, and sets its minimum to :math:`1/(B + 1)`.
#
# One dataset can be a coincidence, so let's estimate how often each rule
# declares a model without signal "different from chance". Under the null
# hypothesis, the observed score is *exchangeable* with the permuted scores
# (this is the very assumption the permutation test relies on). Thus, we can
# simulate many null experiments by drawing an "observed" score and ``B``
# null scores from the pool of 5000 chance-level scores we already have.

# sphinx_gallery_start_ignore


def perm_pval(t_observed, t_null):
    """Two-sided permutation p-value."""
    t_dist = np.abs(t_null - t_null.mean())
    t_obs_dist = np.abs(t_observed - t_null.mean())
    return (np.sum(t_dist >= t_obs_dist) + 1) / (len(t_null) + 1)


def simulate_false_positives(t_perm_scores, n_perms, n_experiments, seed=42):
    """Simulate null experiments and count rejections for each rule."""
    t_rng = np.random.default_rng(seed)
    t_rows = []
    for n_perm in n_perms:
        n_sem, n_pct, n_pval = 0, 0, 0
        for _ in range(n_experiments):
            t_draw = t_rng.choice(
                t_perm_scores, size=n_perm + 1, replace=False
            )
            t_obs, t_null = t_draw[0], t_draw[1:]
            t_sem = t_null.std(ddof=1) / np.sqrt(n_perm)
            n_sem += np.abs(t_obs - t_null.mean()) > 1.96 * t_sem
            t_low, t_high = np.percentile(t_null, [2.5, 97.5])
            n_pct += (t_obs < t_low) or (t_obs > t_high)
            n_pval += perm_pval(t_obs, t_null) <= 0.05
        t_rows.append(
            {
                "B": n_perm,
                "sem": n_sem / n_experiments,
                "pct": n_pct / n_experiments,
                "pval": n_pval / n_experiments,
            }
        )
    return pd.DataFrame(t_rows).set_index("B")


# sphinx_gallery_end_ignore
fpr_perm = simulate_false_positives(
    perm_scores, n_perms=[10, 30, 100, 300, 1000, 3000], n_experiments=2000
)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    fpr_perm.index,
    fpr_perm["sem"],
    marker="o",
    color=BLUE,
    lw=2,
    label="outside mean ± 1.96·SEM",
)
ax.plot(
    fpr_perm.index,
    fpr_perm["pct"],
    marker="s",
    color=ORANGE,
    lw=2,
    label="outside 2.5-97.5 percentile",
)
ax.plot(
    fpr_perm.index,
    fpr_perm["pval"],
    marker="^",
    color=AQUA,
    lw=2,
    ls="--",
    label="permutation p-value ≤ 0.05",
)
ax.axhline(0.05, color="0.4", lw=1, ls=":", label="nominal 5%")
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel("Number of permutations (B)")
ax.set_ylabel("False-positive rate")
ax.set_title("Models without signal declared 'different from chance'")
ax.legend(loc="center right", frameon=False)
fig.tight_layout()
# sphinx_gallery_end_ignore

###############################################################################
# With the SEM rule, the more permutations we run, the more often a model
# without any signal is declared different from chance, approaching 100%.
#
# The percentile rule converges to the nominal 5% as ``B`` grows. With very
# few permutations it is too liberal, because the 2.5th and 97.5th
# percentiles of a handful of scores are poorly estimated. The permutation
# p-value is valid for any ``B``: up to simulation noise, it stays at or
# below the nominal 5%. With few permutations it is conservative, because the
# smallest p-value it can reach is :math:`1/(B + 1)`. In practice, use the
# p-value, with at least a few hundred permutations.
#
# .. note::
#    Here the test is two-sided (the score is "outside" the band). If only
#    the upper edge is used (the score is "above" the band, which is what
#    "better than chance" means), the SEM rule approaches 50% instead of 100%,
#    which is still far from the nominal rate.

###############################################################################
# Part 2: "Better than chance" is a different question
# ----------------------------------------------------
#
# A correctly computed label-permutation test (as in
# :func:`sklearn.model_selection.permutation_test_score`, see Ojala and
# Garriga [#3]_) tests the null hypothesis that ``X`` and ``y`` are
# independent. In other words, it answers: *is the model better than chance?*
#
# This is a low bar. With enough data, almost any model passes it. Let's
# build a large dataset (5000 samples) with a very weak signal.

rng = np.random.default_rng(42)
df_weak = pd.DataFrame(rng.normal(size=(5000, 10)), columns=X_null)
df_weak["y"] = (0.1 * df_weak["f0"] + rng.normal(size=5000) > 0).astype(int)

score_weak, perm_weak, pvalue_weak = permutation_test_score(
    LogisticRegression(),
    df_weak[X_null],
    df_weak["y"],
    cv=cv,
    n_permutations=200,
    random_state=42,
)
print(f"Accuracy: {score_weak:.3f}, p-value: {pvalue_weak:.3f}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(perm_weak, bins=20, color=NULL, label="null (permuted)")
ax.axvline(score_weak, color=INK, lw=2, ls="--", label="observed")
ax.annotate(
    f"accuracy = {score_weak:.3f}\np = {pvalue_weak:.3f}",
    (score_weak, 0.95),
    xycoords=("data", "axes fraction"),
    xytext=(6, 0),
    textcoords="offset points",
    va="top",
)
# Show the whole range from chance to a good model, not just the null
ax.set_xlim(0.45, 0.65)
ax.set_xlabel("Accuracy")
ax.set_ylabel("Number of permutations")
ax.set_title("A very weak model, significantly better than chance")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
# sphinx_gallery_end_ignore

###############################################################################
# The model is "significantly better than chance", with the smallest p-value
# that 200 permutations can give. Yet it is barely above 50% accuracy.
# Whether it is *useful* is a question that no statistical test answers.
#
# Now consider the more common situation: two models, both above chance. We
# generate a dataset with 200 samples and two blocks of five features each,
# ``a*`` and ``b*``. Both blocks are, *by construction*, equally informative
# about ``y``. Model A only uses the ``a*`` features, and model B only uses
# the ``b*`` features. Over repeated samples of the data, both models have
# *exactly* the same expected performance.

# sphinx_gallery_start_ignore


def make_two_blocks(seed, n_samples=200, n_block=5):
    """Generate data with two equally informative blocks of features."""
    t_rng = np.random.default_rng(seed)
    t_x_a = t_rng.normal(size=(n_samples, n_block))
    t_x_b = t_rng.normal(size=(n_samples, n_block))
    signal = 0.7 * (t_x_a.sum(axis=1) + t_x_b.sum(axis=1)) / np.sqrt(n_block)
    t_y = (signal + t_rng.normal(size=n_samples) > 0).astype(int)
    t_names_a = [f"a{i}" for i in range(n_block)]
    t_names_b = [f"b{i}" for i in range(n_block)]
    t_df = pd.DataFrame(np.c_[t_x_a, t_x_b], columns=t_names_a + t_names_b)
    t_df["y"] = t_y
    return t_df, t_names_a, t_names_b


# sphinx_gallery_end_ignore
df_blocks, X_a, X_b = make_two_blocks(seed=0)

score_a, perm_a, pvalue_a = permutation_test_score(
    LogisticRegression(), df_blocks[X_a], df_blocks["y"], cv=cv
)
score_b, perm_b, pvalue_b = permutation_test_score(
    LogisticRegression(), df_blocks[X_b], df_blocks["y"], cv=cv
)
print(f"Model A: accuracy = {score_a:.3f}, p-value = {pvalue_a:.3f}")
print(f"Model B: accuracy = {score_b:.3f}, p-value = {pvalue_b:.3f}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0.3, 0.85, 56)
ax.hist(np.r_[perm_a, perm_b], bins=bins, color=NULL, label="null (permuted)")
for t_name, t_score in [("A", score_a), ("B", score_b)]:
    ax.axvline(t_score, color=INK, lw=2, ls="--")
    ax.annotate(
        f"model {t_name}",
        (t_score, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-4, -4),
        textcoords="offset points",
        ha="right",
        va="top",
    )
ax.set_xlabel("Accuracy")
ax.set_ylabel("Number of permutations")
ax.set_title("Both models are better than chance")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
# sphinx_gallery_end_ignore

###############################################################################
# Both models are far outside of the null distribution (the null
# distributions of both models overlap, so we show them together). Model A
# is more than 10 points ahead of model B on this dataset. Does this mean
# that it is the better model? No, for two reasons:
#
# * Both scores lying outside the null band says nothing about the
#   difference *between* them. It is a test of each model against chance, not
#   of one model against the other.
# * The observed score is itself a noisy CV estimate. Comparing it to a null
#   band ignores its own uncertainty.
#
# To compare two models, we need a test about their *difference*.

###############################################################################
# Part 3: Comparing two models and the dependence between splits
# ---------------------------------------------------------------
#
# To compare models A and B, we evaluate both with the *same* CV splitter, so
# the scores are paired split by split. Following Nadeau and Bengio [#1]_, we
# use *random subsampling*: each split draws a random 20% of the samples as
# test set and trains on the remaining 80%, independently of the other
# splits. In ``scikit-learn``, this is
# :class:`~sklearn.model_selection.StratifiedShuffleSplit`. We use 250
# splits. Before looking at the results, let's look at the splits themselves.
# Each row is one split, each column one sample.

cv_subsampling = StratifiedShuffleSplit(
    n_splits=250, test_size=0.2, random_state=42
)

# sphinx_gallery_start_ignore
t_splits = list(cv_subsampling.split(df_blocks[X_a], df_blocks["y"]))[:10]
t_grid = np.zeros((len(t_splits), len(df_blocks)))
for t_i, (_, t_test) in enumerate(t_splits):
    t_grid[t_i, t_test] = 1
# Sort the samples by their membership in the test sets of the first splits
t_order = np.lexsort(t_grid[::-1])[::-1]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.imshow(
    t_grid[:, t_order],
    aspect="auto",
    cmap=plt.matplotlib.colors.ListedColormap(["#cde2fb", BLUE]),
    interpolation="nearest",
)
ax.set_yticks(range(10), [f"split {i + 1}" for i in range(10)])
ax.set_xticks([])
ax.set_xlabel("Samples (light: training set, dark: test set)")
ax.set_title("First ten splits of a random subsampling CV")
fig.tight_layout()
# sphinx_gallery_end_ignore

###############################################################################
# Every training set uses 80% of the samples, so any two training sets share
# at least 75% of their samples. Test sets overlap too: a sample can be in
# the test set of many splits, or of none. The models trained in each split
# are therefore very similar to each other, and so are their scores. Adding
# splits re-samples the same 200 samples: it adds *splits*, but not *data*.
# Keep this in mind, we will come back to it.
#
# Now let's run both models with ``julearn`` and compare them using the
# corrected ``t-test``.

scores_a = run_cross_validation(
    X=X_a,
    y="y",
    data=df_blocks,
    X_types=X_types,
    model="logit",
    problem_type="classification",
    cv=cv_subsampling,
)
scores_b = run_cross_validation(
    X=X_b,
    y="y",
    data=df_blocks,
    X_types=X_types,
    model="logit",
    problem_type="classification",
    cv=cv_subsampling,
)
scores_a["model"] = "A"
scores_b["model"] = "B"

print(corrected_ttest(scores_a, scores_b))

###############################################################################
# The difference between the two models is not significant. But how does
# this compare to other tests? For different numbers of splits ``J``, we
# compute three tests on the ``J`` split-wise differences of the scores:
#
# * A **naive paired t-test**, which treats the split-wise differences as
#   independent samples.
# * A **sign-flip permutation test** on the split-wise differences. Under the
#   null hypothesis, the sign of each difference is random. We use 10000
#   permutations, so Monte Carlo noise is not an issue.
# * ``julearn``'s **corrected t-test**, the corrected resampled t-test of
#   Nadeau and Bengio [#1]_. It inflates the variance of the mean difference
#   by :math:`(1/J + n_{test}/n_{train})` instead of :math:`1/J`.
#
# We also compute the 95% CI of the mean difference for both t-tests. Since
# :class:`~sklearn.model_selection.StratifiedShuffleSplit` draws its splits
# one after the other from the same random state, the first ``J`` splits of
# a 250-split CV are the same as those of a ``J``-split CV, so we can use
# subsets of the scores we already have.

# sphinx_gallery_start_ignore


def sign_flip_pval(differences, n_perm, seed):
    """Two-sided sign-flip permutation test on paired differences."""
    t_rng = np.random.default_rng(seed)
    signs = t_rng.choice([-1, 1], size=(n_perm, len(differences)))
    null = (signs * differences).mean(axis=1)
    n_extreme = np.sum(np.abs(null) >= np.abs(differences.mean()))
    return (n_extreme + 1) / (n_perm + 1)


def compare_one(t_scores_a, t_scores_b, n_splits, seed=42):
    """Compare two models using the first `n_splits` splits."""
    # Without repeats, julearn numbers the splits in the "fold" column
    t_a = t_scores_a[t_scores_a["fold"] < n_splits]
    t_b = t_scores_b[t_scores_b["fold"] < n_splits]
    diff = t_a["test_score"].values - t_b["test_score"].values
    n_train = t_a["n_train"].iloc[0]
    n_test = t_a["n_test"].iloc[0]
    t_crit = stats.t.ppf(0.975, df=n_splits - 1)
    naive_se = diff.std(ddof=1) * np.sqrt(1 / n_splits)
    corrected_se = diff.std(ddof=1) * np.sqrt(1 / n_splits + n_test / n_train)
    return {
        "n_splits": n_splits,
        "mean_diff": diff.mean(),
        "naive_p": stats.ttest_rel(
            t_a["test_score"], t_b["test_score"]
        ).pvalue,
        "naive_ci": t_crit * naive_se,
        "signflip_p": sign_flip_pval(diff, n_perm=10000, seed=seed),
        "corrected_p": corrected_ttest(t_a, t_b)["p-val"].iloc[0],
        "corrected_ci": t_crit * corrected_se,
    }


def compare_tests(t_scores_a, t_scores_b, n_splits, seed=42):
    """Compare two models for different numbers of splits."""
    return pd.DataFrame(
        [compare_one(t_scores_a, t_scores_b, j, seed) for j in n_splits]
    ).set_index("n_splits")


# sphinx_gallery_end_ignore
all_n_splits = [5, 10, 25, 50, 100, 250]
comparison = compare_tests(scores_a, scores_b, all_n_splits)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(
    comparison.index,
    comparison["naive_p"],
    marker="o",
    color=BLUE,
    lw=2,
    label="naive paired t-test",
)
ax.plot(
    comparison.index,
    comparison["signflip_p"],
    marker="^",
    color=AQUA,
    lw=2,
    ls="--",
    label="sign-flip permutation",
)
ax.plot(
    comparison.index,
    comparison["corrected_p"],
    marker="s",
    color=ORANGE,
    lw=2,
    label="corrected t-test",
)
ax.axhline(0.05, color="0.4", lw=1, ls=":", label="p = 0.05")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-8, 2)
ax.set_xlabel("Number of CV splits (J)")
ax.set_ylabel("p-value")
ax.set_title("p-value (A vs. B)")
ax.legend(loc="lower left", frameon=False)

ax = axes[1]
for t_col, t_color, t_marker, t_shift, t_label in [
    ("naive_ci", BLUE, "o", 0.92, "naive paired t-test"),
    ("corrected_ci", ORANGE, "s", 1.08, "corrected t-test"),
]:
    ax.errorbar(
        comparison.index * t_shift,
        comparison["mean_diff"],
        yerr=comparison[t_col],
        color=t_color,
        marker=t_marker,
        lw=2,
        capsize=4,
        ls="none",
        label=t_label,
    )
ax.axhline(0, color="0.4", lw=1, ls=":", label="no difference")
# Leave room for the legend above the error bars
t_low, t_high = ax.get_ylim()
ax.set_ylim(t_low, t_high + 0.4 * (t_high - t_low))
ax.set_xscale("log")
ax.set_xlabel("Number of CV splits (J)")
ax.set_ylabel("Accuracy difference (A - B)")
ax.set_title("Mean difference and 95% CI")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
# sphinx_gallery_end_ignore

###############################################################################
# As we add splits, the naive t-test and the sign-flip permutation test drift
# towards "significance" (the naive p-value keeps going down, well below the
# bottom of the plot). The sign-flip p-value only stops at
# :math:`1/(10000 + 1)` because that is the smallest p-value 10000
# permutations can give. The naive CI shrinks around the mean difference
# until it no longer includes zero. Yet we did not collect any new data.
#
# This is the dependence between splits we saw above. Adding splits reduces
# the variance due to the *choice of splits*, but not the variance due to
# the *finite dataset*. Both the naive t-test and the sign-flip test treat
# each split as a new independent sample, so they underestimate the
# uncertainty and become anticonservative.
#
# The corrected t-test adds the :math:`n_{test}/n_{train}` term, which does
# not vanish as the number of splits grows. Its p-value stays stable, and
# its CI always includes zero.
#
# Again, one dataset can be a coincidence. Since we know that models A and B
# have the same expected performance, any "significant" difference is a false
# positive. Let's repeat the comparison on 40 independently generated datasets
# and count how often each test rejects the null hypothesis at
# :math:`\alpha = 0.05`.

# sphinx_gallery_start_ignore


def simulate_model_comparisons(n_datasets, n_splits):
    """Compare models A and B on many datasets, count rejections."""
    t_results = []
    for seed in range(n_datasets):
        t_df, t_x_a, t_x_b = make_two_blocks(seed=seed + 1)
        t_cv = StratifiedShuffleSplit(
            n_splits=max(n_splits), test_size=0.2, random_state=seed
        )
        t_scores = []
        for t_name, t_x in [("A", t_x_a), ("B", t_x_b)]:
            t_score = run_cross_validation(
                X=t_x,
                y="y",
                data=t_df,
                X_types=X_types,
                model="logit",
                problem_type="classification",
                cv=t_cv,
            )
            t_score["model"] = t_name
            t_scores.append(t_score)
        t_results.append(compare_tests(*t_scores, n_splits, seed=seed))
    return (
        pd.concat(t_results)
        .groupby("n_splits")[["naive_p", "signflip_p", "corrected_p"]]
        .apply(lambda x: (x < 0.05).mean())
    )


# sphinx_gallery_end_ignore
fpr_cv = simulate_model_comparisons(n_datasets=40, n_splits=all_n_splits)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    fpr_cv.index,
    fpr_cv["naive_p"],
    marker="o",
    color=BLUE,
    lw=2,
    label="naive paired t-test",
)
ax.plot(
    fpr_cv.index,
    fpr_cv["signflip_p"],
    marker="^",
    color=AQUA,
    lw=2,
    ls="--",
    label="sign-flip permutation",
)
ax.plot(
    fpr_cv.index,
    fpr_cv["corrected_p"],
    marker="s",
    color=ORANGE,
    lw=2,
    label="corrected t-test",
)
ax.axhline(0.05, color="0.4", lw=1, ls=":", label="nominal 5%")
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel("Number of CV splits (J)")
ax.set_ylabel("False-positive rate")
ax.set_title("Two equivalent models declared different")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
# sphinx_gallery_end_ignore

###############################################################################
# The naive t-test and the sign-flip permutation test declare a difference
# between two equivalent models more and more often as we add splits. (With
# only 5 splits, the sign-flip test never rejects: there are only
# :math:`2^5 = 32` possible sign flips, so its smallest p-value is
# :math:`2/32 > 0.05`.) The corrected t-test stays close to the nominal 5%,
# regardless of the number of splits. With 40 datasets, each false-positive
# rate is only estimated within about :math:`\pm 7` percentage points, so the
# differences between 2.5% and 10% are noise. The empirical behavior of these
# tests, including how *replicable* their conclusions are, was studied in
# detail by Bouckaert and Frank [#2]_.
#
# So the problem is not *t-test vs. permutation test*. The problem is
# ignoring the dependence between splits, and a permutation test on
# split-wise scores makes the same mistake as the naive t-test.
#
# Take-home messages
# ------------------
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 35 35
#
#    * - Question
#      - Use
#      - Do not use
#    * - Is my model better than chance?
#      - Label-permutation p-value, e.g.
#        :func:`sklearn.model_selection.permutation_test_score`, with at
#        least a few hundred permutations.
#      - A "CI of the permuted model" built with the SEM.
#    * - Is model A better than model B?
#      - The corrected t-test, :func:`.corrected_ttest`, with both models
#        evaluated on the same CV splits.
#      - A naive t-test or a permutation test on the split-wise scores.
#        Two models being "better than chance".
#
# * Passing the "better than chance" test does not mean that the model is
#   useful.
# * Neither the number of permutations nor the number of CV splits should be
#   able to change your conclusion. If it does, the test is wrong.
#
# .. topic:: References:
#
#   .. [#1] Nadeau and Bengio, `"Inference for the Generalization Error" \
#      <https://doi.org/10.1023/A:1024068626366>`_, Machine Learning,
#      Volume 52, 2003, pages 239-281.
#
#   .. [#2] Bouckaert and Frank, `"Evaluating the Replicability of \
#      Significance Tests for Comparing Learning Algorithms" \
#      <https://doi.org/10.1007/978-3-540-24775-3_3>`_, PAKDD 2004, Lecture
#      Notes in Computer Science, Volume 3056, pages 3-12.
#
#   .. [#3] Ojala and Garriga, `"Permutation Tests for Studying Classifier \
#      Performance" <https://jmlr.org/papers/v11/ojala10a.html>`_, JMLR,
#      Volume 11, 2010, pages 1833-1863.
