# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from julearn.transformers import DropColumns

X = pd.DataFrame(
    dict(A=np.arange(10), B=np.arange(10, 20), C=np.arange(30, 40))
)

X_with_types = pd.DataFrame(
    {
        "a__:type:__continuous": np.arange(10),
        "b__:type:__continuous": np.arange(10, 20),
        "c__:type:__confound": np.arange(30, 40),
        "d__:type:__confound": np.arange(40, 50),
        "e__:type:__categorical": np.arange(40, 50),
        "f__:type:__categorical": np.arange(40, 50),
    }
)


def test_DropColumns():
    drop_columns = DropColumns(columns=".*__:type:__confound")
    X_trans = drop_columns.fit_transform(X_with_types)

    kept_cols = X_with_types.columns[drop_columns.get_support()].to_list()
    kept_cols_2 = X_with_types.iloc[
        :, drop_columns.get_support(True)
    ].columns.to_list()
    assert_frame_equal(X_trans, X_with_types[kept_cols])
    assert_array_equal(X_trans, X_with_types[kept_cols_2])
    assert_frame_equal(
        X_with_types.drop(
            columns=["c__:type:__confound", "d__:type:__confound"]
        ),
        X_trans,
    )
