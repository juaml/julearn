"""Provides tests for the FilterColumns class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from julearn.transformers.dataframe import FilterColumns


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


def test_FilterColumns() -> None:
    """Test FilterColumns."""
    filter = FilterColumns(keep=["continuous"])
    kept_columns = [
        "a__:type:__continuous",
        "b__:type:__continuous",
    ]
    filter.set_output(transform="pandas").fit(X_with_types)
    X_expected = X_with_types.copy()[kept_columns]
    X_trans = filter.transform(X_with_types)
    assert isinstance(X_expected, pd.DataFrame)
    assert_frame_equal(X_expected, X_trans)
