"""Provides tests for the DropColumns class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from julearn.transformers.dataframe import DropColumns


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


def test_DropColumns() -> None:
    """Test DropColumns."""
    drop_columns = DropColumns(apply_to=["confound"])
    drop_columns.fit(X_with_types)
    X_trans = drop_columns.transform(X_with_types)
    support = drop_columns.get_support()

    non_confound = [
        "a__:type:__continuous",
        "b__:type:__continuous",
        "e__:type:__categorical",
        "f__:type:__categorical",
    ]

    X_non_confound = X_with_types[non_confound]
    assert_frame_equal(X_trans, X_non_confound)
    assert_frame_equal(
        X_with_types.drop(
            columns=["c__:type:__confound", "d__:type:__confound"]
        ),
        X_trans,
    )
    assert all(support == [1, 1, 0, 0, 1, 1])
