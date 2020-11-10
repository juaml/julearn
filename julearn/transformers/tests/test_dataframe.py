# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from julearn.transformers import DataFrameTransformer
from julearn.utils.testing import PassThroughTransformer

X = pd.DataFrame(dict(A=np.arange(10),
                      B=np.arange(10, 20),
                      C=np.arange(30, 40)
                      ))

X_with_types = pd.DataFrame({
    'a__:type:__continuous': np.arange(10),
    'b__:type:__continuous': np.arange(10, 20),
    'c__:type:__confound': np.arange(30, 40),
    'd__:type:__confound': np.arange(40, 50),
    'e__:type:__categorical': np.arange(40, 50),
    'f__:type:__categorical': np.arange(40, 50),
})


def test_transform_all_return_same_passthrough():

    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='all',
                                    returned_features='same',
                                    )

    X_trans = trans_df.fit_transform(X)
    assert_frame_equal(X_trans, X)


def test_all_return_unknown_passthrough():

    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='all',
                                    returned_features='unknown',
                                    )

    X_trans = trans_df.fit_transform(X)
    X_trans_same = X_trans.copy()
    X_trans_same.columns = X.columns
    assert_array_equal(X_trans.values, X.values)
    assert_frame_equal(X_trans_same, X)


def test_pca_transform_all():
    trans_df = DataFrameTransformer(transformer=PCA(n_components=2,
                                                    random_state=1),
                                    transform_column='all',
                                    returned_features='unknown',
                                    )

    trans = PCA(n_components=2,
                random_state=1)

    X_df_transformed = trans_df.fit_transform(X)
    # internally we use fit().transform() for fit_transform(),
    # but on the PCA it self fit_transform() and fit().transform()
    # do not result in the same matrix
    X_transformed = trans.fit(X).transform(X)

    assert (len(X_df_transformed.columns)
            == len(X_transformed[0])
            == 2)
    assert_array_equal(X_df_transformed.values,
                       X_transformed)


def test_pca_transform_AB():

    trans_df = DataFrameTransformer(transformer=PCA(n_components=1,
                                                    random_state=1),
                                    transform_column=['A', 'B'],
                                    returned_features='unknown',
                                    )

    X_df_transformed = trans_df.fit_transform(X)
    trans = PCA(n_components=1,
                random_state=1)
    X_transformed = (trans.fit(X.loc[:, ['A', 'B']])
                          .transform(X.loc[:, ['A', 'B']]))

    assert_array_equal(X_df_transformed
                       .drop(columns='C').values, X_transformed)
    assert_array_equal(X_df_transformed.C.values, X.C.values)


def test_pca_transform_continuous_return_unknown():

    trans_df = DataFrameTransformer(transformer=PCA(n_components=1,
                                                    random_state=1),
                                    transform_column='continuous',
                                    returned_features='unknown',
                                    )
    trans_df.fit(X)
    assert (trans_df.transform_column_ == X.columns).all()


def test_scale_columns_of_type_return_same():
    all_columns = list(X_with_types.columns)

    condition_columns = (
        ('all', all_columns),
        ('all_features', all_columns[:2] + all_columns[4:]),
        ('continuous', all_columns[:2]),
        ('categorical', all_columns[4:]),
        ('confound', all_columns[2:4]),
        (['continuous', 'categorical'], all_columns[:2] + all_columns[4:]),
        (['confound', 'continuous'], all_columns[:4]),
        (all_columns[1:3], all_columns[1:3]),
    )

    for condition, columns in condition_columns:
        trans_df = DataFrameTransformer(transformer=StandardScaler(),
                                        transform_column=condition,
                                        returned_features='same',
                                        )

        np.random.seed(42)
        X_trans_df = trans_df.fit(X_with_types).transform(X_with_types)

        np.random.seed(42)
        X_trans = StandardScaler().fit_transform(X_with_types.loc[:, columns])
        X_trans = pd.DataFrame(X_trans,
                               columns=columns,
                               index=X_with_types.copy().index)
        if len(X_trans.columns) != len(X_with_types.columns):
            X_rest = X_with_types.copy().drop(columns=columns)
            X_trans = pd.concat([X_trans, X_rest], axis=1).reindex(
                columns=X_with_types.columns)

        assert_frame_equal(X_trans, X_trans_df)


def test_pca_columns_of_type_return_unknown_or_unknown_same_type():
    all_columns = list(X_with_types.columns)

    condition_columns = (
        ('all', all_columns),
        ('all_features', all_columns[:2] + all_columns[4:]),
        ('continuous', all_columns[:2]),
        ('categorical', all_columns[4:]),
        ('confound', all_columns[2:4]),
        (['continuous', 'categorical'], all_columns[:2] + all_columns[4:]),
        (['confound', 'continuous'], all_columns[:4]),
        (all_columns[1:3], all_columns[1:3]),
    )

    for condition, columns in condition_columns:
        for returned_features in ['unknown', 'unknown_same_type']:
            trans_df = DataFrameTransformer(
                transformer=PCA(),
                transform_column=condition,
                returned_features=returned_features,
            )

            np.random.seed(42)
            if (returned_features == 'unknown_same_type') and (
                (type(condition) == list) or (condition in [
                    'all', 'all_features'])):
                with pytest.raises(ValueError,
                                   match=r'You can only return same type, '):
                    trans_df.fit(X_with_types).transform(X_with_types)
                continue

            X_trans_df = trans_df.fit(X_with_types).transform(X_with_types)

            np.random.seed(42)
            X_trans = (PCA()
                       .fit(X_with_types.loc[:, columns])
                       .transform(X_with_types.loc[:, columns])
                       )
            X_trans = pd.DataFrame(X_trans,
                                   columns=columns,
                                   index=X_with_types.copy().index)

            if len(X_trans.columns) != len(X_with_types.columns):
                X_rest = X_with_types.copy().drop(columns=columns)
                X_trans = pd.concat([X_trans, X_rest], axis=1)
            assert_array_equal(X_trans.values, X_trans_df.values)


def test_error_returned_feature_input():
    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='all',
                                    returned_features='WrongInput',
                                    )

    with pytest.raises(ValueError, match='returned_features can only be'):
        trans_df.fit_transform(X)


def test_error_no_matching_transform_column():

    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='confound',
                                    returned_features='same',
                                    )

    with pytest.raises(ValueError,
                       match='There is not valid column to transform '):
        trans_df.fit_transform(X)


def test_error_returned_features_subset():

    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='confound',
                                    returned_features='subset',
                                    )

    with pytest.raises(ValueError,
                       match='You cannot use subset on a transformer'):
        trans_df.fit_transform(X_with_types)
