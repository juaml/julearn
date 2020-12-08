# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler

from julearn.transformers import (DataFrameWrapTransformer,
                                  ChangeColumnTypes,
                                  DropColumns)
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


def test_error_no_matching_transform_column():

    trans_df = DataFrameWrapTransformer(transformer=StandardScaler(),
                                        apply_to='confound',
                                        returned_features='same',
                                        )

    with pytest.raises(ValueError,
                       match='There is not valid column to transform '):
        trans_df.fit_transform(X)


def test_error_returned_features_subset():

    trans_df = DataFrameWrapTransformer(transformer=StandardScaler(),
                                        apply_to='confound',
                                        returned_features='subset',
                                        )

    with pytest.raises(ValueError,
                       match='You can only subset with a '):
        trans_df.fit_transform(X_with_types)


def test_ChangeColumnTypes():
    X_with_types_copy = X_with_types.copy()
    X_with_types_changed = X_with_types_copy.copy()
    X_with_types_changed.columns = ['a__:type:__continuous',
                                    'b__:type:__continuous',
                                    'c__:type:__continuous',
                                    'd__:type:__continuous',
                                    'e__:type:__categorical',
                                    'f__:type:__categorical']
    scales_both = DataFrameWrapTransformer(
        transformer=StandardScaler(),
        returned_features='same',
        apply_to=['continuous', 'confound'])
    scales_continuous = DataFrameWrapTransformer(
        transformer=StandardScaler(),
        returned_features='same',
        apply_to=['continuous'])
    change_types = DataFrameWrapTransformer(
        ChangeColumnTypes(columns_match='.*confound', new_type='continuous'),
        returned_features='from_transformer'
    )

    X_trans = change_types.fit_transform(X_with_types_copy)
    X_scaled_both = scales_both.fit_transform(X_with_types_copy)
    X_scaled_trans = scales_continuous.fit_transform(X_trans)

    assert_frame_equal(
        X_trans.reindex(X_trans.columns.sort_values()),
        X_with_types_changed.reindex(
            X_with_types_changed.columns.sort_values()
        )
    )
    assert_array_equal(X_scaled_both.values, X_scaled_trans.values)


def test_DropColumns():
    drop_columns = DropColumns(columns='.*__:type:__confound')
    X_trans = drop_columns.fit_transform(X_with_types)

    kept_cols = X_with_types.columns[drop_columns.get_support()].to_list()
    kept_cols_2 = (X_with_types
                   .iloc[:, drop_columns.get_support(True)].columns.to_list())
    assert_frame_equal(
        X_trans,
        X_with_types[kept_cols]
    )
    assert_array_equal(
        X_trans,
        X_with_types[kept_cols_2]
    )
    assert_frame_equal(
        X_with_types.drop(
            columns=['c__:type:__confound', 'd__:type:__confound']),
        X_trans)
