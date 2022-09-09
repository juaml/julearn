# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import pearsonr, spearmanr
from seaborn import load_dataset
from julearn.transformers import CBPM

df_iris = load_dataset('iris')

X_pos = ['sepal_length', 'petal_length', 'petal_width']
X_neg = ['sepal_width']
X_posneg = X_pos + X_neg

y = 'species'
df_iris[y] = df_iris[y].apply(
    lambda x: {'setosa': 0, 'versicolor': 1, 'virginica': 3}[x])

df_shuffled_X = df_iris.copy()
df_shuffled_X[X_posneg] = (df_shuffled_X[X_posneg]
                           .sample(frac=1, random_state=42)
                           .values
                           )


def test_posneg_correlated_features():
    trans_X_posneg = (CBPM(corr_sign='posneg', agg_method=np.mean)
                      .fit_transform(df_iris[X_posneg], df_iris[y])
                      )
    trans_man_pos = df_iris[X_pos].values.mean(axis=1)
    trans_man_neg = df_iris[X_neg].values.mean(axis=1)
    trans_man = np.concatenate([trans_man_pos.reshape(-1, 1),
                                trans_man_neg.reshape(-1, 1)], axis=1)
    assert_array_equal(trans_X_posneg, trans_man)


def test_pos_correlated_features():
    trans_X_pos = (CBPM(corr_sign='pos', agg_method=np.mean)
                   .fit_transform(df_iris[X_pos], df_iris[y])
                   )

    trans_X_pos_neg = (CBPM(corr_sign='pos', agg_method=np.mean)
                       .fit_transform(df_iris[X_posneg], df_iris[y])
                       )

    trans_man = df_iris[X_pos].values.mean(axis=1)

    assert_array_equal(trans_X_pos, trans_X_pos_neg)
    assert_array_equal(trans_X_pos, trans_man)


def test_neg_correlated_features():
    trans_X_neg = (CBPM(corr_sign='neg', agg_method=np.mean)
                   .fit_transform(df_iris[X_neg], df_iris[y])
                   )

    trans_X_pos_neg = (CBPM(corr_sign='neg', agg_method=np.mean)
                       .fit_transform(df_iris[X_posneg], df_iris[y])
                       )

    trans_man = df_iris[X_neg].values.mean(axis=1)

    assert_array_equal(trans_X_neg, trans_X_pos_neg)
    assert_array_equal(trans_X_neg, trans_man)


def test_warn_pos_no_feat():
    with pytest.warns(RuntimeWarning,
                      match='No feature is significant'):
        trans = (CBPM(corr_sign='pos', agg_method=np.mean)
                 .fit_transform(df_iris[X_neg], df_iris[y])
                 )

    assert (trans == df_iris[y].values.mean()).all()


def test_warn_neg_no_feat():
    with pytest.warns(RuntimeWarning,
                      match='No feature is significant'):
        trans = (CBPM(corr_sign='neg', agg_method=np.mean)
                 .fit_transform(df_iris[X_pos], df_iris[y])
                 )

    assert (trans == df_iris[y].values.mean()).all()


def test_warn_posneg_no_feat():
    with pytest.warns(RuntimeWarning,
                      match='No feature is significant'):
        trans = (CBPM(corr_sign='posneg', agg_method=np.mean)
                 .fit_transform(df_shuffled_X[X_posneg], df_iris[y])
                 )
    assert (trans == df_iris[y].values.mean()).all()


def test_warn_posneg_no_pos_feat():
    with pytest.warns(RuntimeWarning,
                      match='No feature with significant positive'):
        trans_posneg = (CBPM(corr_sign='posneg', agg_method=np.mean)
                        .fit_transform(df_iris[X_neg], df_iris[y])
                        )

    trans_man = df_iris[X_neg].values.mean(axis=1)

    assert_array_equal(trans_posneg, trans_man)


def test_warn_posneg_no_neg_feat():
    with pytest.warns(RuntimeWarning,
                      match='No feature with significant negative'):
        trans_posneg = (CBPM(corr_sign='posneg', agg_method=np.mean)
                        .fit_transform(df_iris[X_pos], df_iris[y])
                        )

    trans_man = df_iris[X_pos].values.mean(axis=1)

    assert_array_equal(trans_posneg, trans_man)


def test_lower_sign_threshhold():

    # I have checked before that only these 2 have pvalues under 1e-50
    trans_posneg = (CBPM(corr_sign='pos',
                         significance_threshold=1e-50,
                         agg_method=np.mean)
                    .fit_transform(df_iris[X_pos], df_iris[y])
                    )
    trans_man = df_iris[['petal_length', 'petal_width']].values.mean(axis=1)

    assert_array_equal(trans_posneg, trans_man)


def test_lower_sign_threshhold_no_sig():

    # I have checked before there are no under 1e-100
    with pytest.warns(RuntimeWarning,
                      match='No feature is significant'):
        trans_posneg = (CBPM(corr_sign='pos',
                             significance_threshold=1e-100,
                             agg_method=np.mean
                             )
                        .fit_transform(df_iris[X_pos], df_iris[y])
                        )
    assert (trans_posneg == df_iris[y].values.mean()).all()


def test_spearman():

    # I have checked before all are still sign with spearman
    trans_posneg = (CBPM(corr_method=spearmanr, agg_method=np.mean)
                    .fit_transform(df_iris[X_posneg], df_iris[y])
                    )

    trans_man_pos = df_iris[X_pos].values.mean(axis=1)
    trans_man_neg = df_iris[X_neg].values.mean(axis=1)
    trans_man = np.concatenate([trans_man_pos.reshape(-1, 1),
                                trans_man_neg.reshape(-1, 1)], axis=1)
    assert_array_equal(trans_posneg, trans_man)
