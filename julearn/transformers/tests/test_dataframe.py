# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.decomposition import PCA

from julearn.transformers import DataFrameTransformer
from julearn.transformers.basic import PassThroughTransformer


X = pd.DataFrame(dict(A=np.arange(10),
                      B=np.arange(10, 20),
                      C=np.arange(30, 40)
                      ))


def test_all_DataFrameTransformer():

    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='all',
                                    returned_features='same',
                                    )

    X_trans = trans_df.fit_transform(X)
    assert_frame_equal(X_trans, X)


def test_all_unknown_DataFrameTransformer():

    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='all',
                                    returned_features='unknown',
                                    )

    X_trans = trans_df.fit_transform(X)
    X_trans_same = X_trans.copy()
    X_trans_same.columns = X.columns
    assert_array_equal(X_trans.values, X.values)
    assert_frame_equal(X_trans_same, X)


def test_pca_DataFrameTransformer():
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
    # do not resutl in the same matrix
    X_transformed = trans.fit(X).transform(X)

    assert (len(X_df_transformed.columns)
            == len(X_transformed[0])
            == 2)
    assert_array_equal(X_df_transformed.values,
                       X_transformed)


def test_pca_on_some_columns_DataFrameTransformer():

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


def test_set_columns_to_transform_DataFrameTransformer():

    trans_df = DataFrameTransformer(transformer=PCA(n_components=1,
                                                    random_state=1),
                                    transform_column='continuous',
                                    returned_features='unknown',
                                    )
    trans_df.fit(X)
    assert (trans_df.transform_column_ == X.columns).all()


def test_get_columns_of_type_DataFrameTransformer():
    columns = ['a__:type:__continouse',
               'b__:type:__continouse',
               'conf__:type:__confound']
    trans_df = DataFrameTransformer(transformer=PassThroughTransformer(),
                                    transform_column='all',
                                    returned_features='unknown',
                                    )

    trans_df.fit(X)
    assert (trans_df.get_columns_of_type(columns, 'confound')
            == ['conf__:type:__confound'])
