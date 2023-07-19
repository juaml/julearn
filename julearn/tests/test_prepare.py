"""Provide tests for the prepare module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from julearn.model_selection import (
    RepeatedContinuousStratifiedGroupKFold,
    ContinuousStratifiedGroupKFold,
)
from julearn.prepare import (
    _check_x_types,
    _pick_columns,
    check_consistency,
    prepare_input_data,
)
from julearn.config import set_config


def _check_df_input(prepared, X, y, groups, df):
    df_X, df_y, df_groups, _ = prepared

    assert_array_equal(df[X].values, df_X[X].values)
    assert_array_equal(df_y.values, df[y].values)
    if groups is not None:
        assert_array_equal(df[groups].values, df_groups)


def test_prepare_input_data() -> None:
    """Test prepare input data (dataframe)."""
    data = np.random.rand(4, 10)
    columns = [f"f_{x}" for x in range(data.shape[1])]

    # Test X (2d) + y
    X = columns[:-2]
    y = columns[-1]
    df = pd.DataFrame(data=data, columns=columns)
    X_types = {"continuous": X}

    prepared = prepare_input_data(
        X=X, y=y, df=df, pos_labels=None, groups=None, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=None, df=df)

    # Test X (2d) + y + groups
    X = columns[:5]
    y = columns[7]
    groups = columns[8]
    X_types = {"continuous": X}
    prepared = prepare_input_data(
        X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # Test X (2d) + groups
    X = columns[:5]
    y = columns[6]
    groups = columns[7]
    X_types = {"continuous": X}
    prepared = prepare_input_data(
        X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # Test X (1d) + y + groups + confounds (1d)
    X = columns[2]
    y = columns[6]
    groups = columns[7]
    X_types = {"continuous": X}
    prepared = prepare_input_data(
        X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # Test using [":"] to select all columns in X (But y)
    X = columns[:-1]
    y = columns[-1]
    groups = None
    X_types = {"continuous": X}
    prepared = prepare_input_data(
        X=[":"], y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # Test using [":"] to select all columns in X (But y and groups)
    X = columns[:-2]
    y = columns[-2]
    groups = columns[-1]
    X_types = {"continuous": X}
    prepared = prepare_input_data(
        X=[":"], y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # Test picks using regexp
    columns = [f"f_{x}" for x in range(data.shape[1] - 2)]
    columns.append("t_8")
    columns.append("g_9")
    df = pd.DataFrame(data=data, columns=columns)

    X = columns[:-2]
    y = "t_8"
    groups = "g_9"
    X_types = {"continuous": X}
    prepared = prepare_input_data(
        X="f_.*", y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    prepared = prepare_input_data(
        X=["f_.*"], y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # Now use regular expressions for X_types too
    X_types = {"continuous": "f_.*"}
    prepared = prepare_input_data(
        X=["f_.*"], y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    X_types = {"continuous": ["f_.*"]}
    prepared = prepare_input_data(
        X=["f_.*"], y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    X_types = {"continuous": ["f_.*"]}
    prepared = prepare_input_data(
        X="f_.*", y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    X_types = {"continuous": "f_.*"}
    prepared = prepare_input_data(
        X="f_.*", y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)


def test_prepare_input_data_erors() -> None:
    """Test prepare input data (dataframe) errors."""
    data = np.random.rand(4, 10)
    columns = [f"f_{x}" for x in range(data.shape[1])]
    df = pd.DataFrame(data=data, columns=columns)

    # Wrong types for dataframe columns
    with pytest.raises(ValueError, match=r"DataFrame columns must be strings"):
        X = 2
        y = columns[6]
        data = np.random.rand(4, 10)
        int_columns = [f"f_{x}" for x in range(data.shape[1] - 1)] + [0]

        X = columns[:-2]
        y = columns[-1]
        df_wrong_cols = pd.DataFrame(data=data, columns=int_columns)

        prepared = prepare_input_data(
            X=X,
            y=y,
            df=df_wrong_cols,
            pos_labels=None,
            groups=None,
            X_types=None,
        )

    # Wrong types for X
    with pytest.raises(
        ValueError, match=r"X must be a string or list of strings"
    ):
        X = 2
        y = columns[6]
        prepared = prepare_input_data(
            X=X,  # type: ignore
            y=y,
            df=df,
            pos_labels=None,
            groups=None,
            X_types=None,  # type: ignore
        )

    # Wrong types for y
    with pytest.raises(ValueError, match=r"y must be a string"):
        X = columns[:5]
        y = ["bad"]
        prepared = prepare_input_data(
            X=X,
            y=y,  # type: ignore
            df=df,
            pos_labels=None,
            groups=None,
            X_types=None,  # type: ignore
        )

    # Wrong types for groups
    with pytest.raises(ValueError, match=r"groups must be a string"):
        X = columns[:5]
        y = columns[6]
        groups = 2
        prepared = prepare_input_data(
            X=X,
            y=y,
            df=df,
            pos_labels=None,
            groups=groups,  # type: ignore
            X_types=None,  # type: ignore
        )

    # Wrong types for df
    with pytest.raises(ValueError, match=r"df must be a pandas.DataFrame"):
        X = columns[:5]
        y = columns[6]
        prepared = prepare_input_data(
            X=X,
            y=y,
            df=dict(),  # type: ignore
            pos_labels=None,
            groups=None,
            X_types=None,  # type: ignore
        )

    # Missing column in dataframe
    X = columns[:5] + ["wrong"]
    y = columns[6]
    groups = columns[7]
    with pytest.raises(ValueError, match=r"missing: \['wrong'\]"):
        prepared = prepare_input_data(
            X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=None
        )

    # Disable X check should not raise an error
    set_config("disable_x_check", True)
    prepared = prepare_input_data(
        X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=None
    )
    set_config("disable_x_check", False)

    # Missing target in dataframe
    X = columns[:5]
    y = "wrong"
    groups = columns[7]
    with pytest.raises(ValueError, match=r"not a valid column"):
        prepared = prepare_input_data(
            X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=None
        )

    # Missing groups in dataframe
    X = columns[:5]
    y = columns[6]
    groups = "wrong"
    with pytest.raises(ValueError, match=r"is not a valid column"):
        prepared = prepare_input_data(
            X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=None
        )

    # Test overlapping X, y and groups

    # y in X
    X = columns[:5]
    y = columns[4]
    groups = columns[7]
    with pytest.warns(RuntimeWarning, match="contains the target"):
        prepared = prepare_input_data(
            X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=None
        )
    _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # y and groups
    X = columns[:5]
    y = columns[6]
    groups = columns[6]
    with pytest.warns(
        RuntimeWarning, match="y and groups are the same column"
    ):
        prepared = prepare_input_data(
            X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=None
        )
        _check_df_input(prepared, X=X, y=y, groups=groups, df=df)

    # X and groups
    X = columns[:5]
    y = columns[6]
    groups = columns[3]
    with pytest.warns(RuntimeWarning, match="groups is part of X"):
        prepared = prepare_input_data(
            X=X, y=y, df=df, pos_labels=None, groups=groups, X_types=None
        )
        _check_df_input(prepared, X=X, y=y, groups=groups, df=df)


def test_prepare_input_data_pos_labels() -> None:
    """Test prepare input data (dataframe) pos_labels."""
    data = np.random.rand(20, 10)
    columns = [f"f_{x}" for x in range(data.shape[1])]
    df = pd.DataFrame(data=data, columns=columns)
    X = columns[:-1]
    y = columns[-1]
    X_types = {"continuous": X}
    # Test pos_labels as int
    t_df = df.copy()
    t_df[y] = (t_df[y] > 0.5).astype(int)
    _, prep_y, _, _ = prepare_input_data(
        X=X, y=y, df=t_df, pos_labels=1, groups=None, X_types=X_types
    )
    assert_series_equal(prep_y, t_df[y])

    _, prep_y, _, _ = prepare_input_data(
        X=X, y=y, df=t_df, pos_labels=0, groups=None, X_types=X_types
    )
    assert_series_equal(prep_y, 1 - t_df[y])

    # Test pos labels as str
    t_df = df.copy()
    t_df[y] = "mid"
    target = t_df[y]
    high_mask = df[y] > 0.8
    low_mask = df[y] < 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target.loc[high_mask] = "high"
        target.loc[low_mask] = "low"
    t_df[y] = target

    bin_y = (target == "high").astype(int)
    assert 0 in bin_y.values
    assert 1 in bin_y.values
    _, prep_y, _, _ = prepare_input_data(
        X=X, y=y, df=t_df, pos_labels="high", groups=None, X_types=X_types
    )
    assert_series_equal(prep_y, bin_y)

    bin_y = (target == "low").astype(int)
    assert 0 in bin_y.values
    assert 1 in bin_y.values
    _, prep_y, _, _ = prepare_input_data(
        X=X, y=y, df=t_df, pos_labels="low", groups=None, X_types=X_types
    )
    assert_series_equal(prep_y, bin_y)

    bin_y = (target == "mid").astype(int)
    assert 0 in bin_y.values
    assert 1 in bin_y.values
    _, prep_y, _, _ = prepare_input_data(
        X=X, y=y, df=t_df, pos_labels="mid", groups=None, X_types=X_types
    )
    assert_series_equal(prep_y, bin_y)

    bin_y = target.isin(["low", "mid"]).astype(int)
    assert 0 in bin_y.values
    assert 1 in bin_y.values
    _, prep_y, _, _ = prepare_input_data(
        X=X,
        y=y,
        df=t_df,
        pos_labels=["low", "mid"],
        groups=None,
        X_types=X_types,
    )
    assert_series_equal(prep_y, bin_y)

    with pytest.warns(RuntimeWarning, match="labels are not in the target"):
        bin_y = (target == "low").astype(int)
        assert 0 in bin_y.values
        assert 1 in bin_y.values
        _, prep_y, _, _ = prepare_input_data(
            X=X,
            y=y,
            df=t_df,
            pos_labels=["low", "missing"],
            groups=None,
            X_types=X_types,
        )
        assert_series_equal(prep_y, bin_y)

    with pytest.warns(RuntimeWarning, match="All targets have been set to 1"):
        bin_y = target.isin(["low", "mid", "high"]).astype(int)
        assert 0 not in bin_y.values
        assert 1 in bin_y.values
        _, prep_y, _, _ = prepare_input_data(
            X=X,
            y=y,
            df=t_df,
            pos_labels=["low", "mid", "high"],
            groups=None,
            X_types=X_types,
        )
        assert_series_equal(prep_y, bin_y)

    with pytest.warns(RuntimeWarning, match="All targets have been set to 0"):
        bin_y = target.isin(["wrong"]).astype(int)
        assert 0 in bin_y.values
        assert 1 not in bin_y.values
        _, prep_y, _, _ = prepare_input_data(
            X=X,
            y=y,
            df=t_df,
            pos_labels=["wrong"],
            groups=None,
            X_types=X_types,
        )
        assert_series_equal(prep_y, bin_y)


def test_pick_columns_using_column_name() -> None:
    """Test pick columns using column names as regexes."""
    columns = ["conf_1", "conf_2", "feat_1", "feat_2", "Feat_3"]
    regexes = ["conf_2", "Feat_3"]

    assert regexes == _pick_columns(regexes, columns)

    columns = ["Feat_3", "conf_1", "conf_2", "feat_1", "feat_2"]
    regexes = ["conf_2", "Feat_3"]

    assert regexes == _pick_columns(regexes, columns)

    columns = ["120", "121", "122", "123", "124", "125"]
    regexes = ["12"]
    msg = r"following are missing: \['12'\]"
    with pytest.raises(ValueError, match=msg):
        _pick_columns(regexes, columns)

    columns = ["120", "121", "122", "123", "124", "125"]
    regexes = ["2"]
    msg = r"following are missing: \['2'\]"
    with pytest.raises(ValueError, match=msg):
        _pick_columns(regexes, columns)

    columns = ["120", "121", "122", "123", "124", "125"]
    regexes = ["24"]
    msg = r"following are missing: \['24'\]"
    with pytest.raises(ValueError, match=msg):
        _pick_columns(regexes, columns)

    columns = ["120", "121", "122", "123", "124", "125"]
    regexes = ["122", "125", "130"]
    msg = r"following are missing: \['130'\]"
    with pytest.raises(ValueError, match=msg):
        _pick_columns(regexes, columns)

    columns = ["120", "121", "122", "123", "124", "125"]
    regexes = ["122", "125"]
    assert regexes == _pick_columns(regexes, columns)


def test_pick_columns_using_regex_match() -> None:
    """Test pick columns using regexes."""
    columns = ["conf_1", "conf_2", "feat_1", "feat_2", "Feat_3"]
    regexes = [".*conf.*", ".*feat.*"]

    picked = _pick_columns(regexes, columns)
    assert columns[:-1] == picked

    # Test with overlapping/repeated regexes
    columns = ["conf_1", "conf_2", "_feat_1", "feat_2", "Feat_3"]
    regexes = [".*conf.*", ".*feat.*", "feat_.*"]

    picked = _pick_columns(regexes, columns)
    assert columns[:-1] == picked


def test_pick_columns_using_regex_and_column_name_match() -> None:
    """Test pick columns using regexes and column names."""
    columns = ["conf_1", "conf_2", "feat_1", "feat_2", "Feat_3"]
    regexes = [".*conf.*", ".*feat.*", "Feat_3"]

    assert columns == _pick_columns(regexes, columns)


def test_prepare_data_pick_regexp():
    """Test picking columns by regexp."""
    data = np.random.rand(5, 10)
    columns = [
        "_a_b_c1_",
        "_a_b_c2_",
        "_a_b2_c3_",
        "_a_b2_c4_",
        "_a_b3_c5_",
        "_a_b3_c6_",
        "_a3_b2_c7_",
        "_a2_b_c7_",
        "_a2_b_c8_",
        "_a2_b_c9_",
    ]

    X = columns[:-1]
    y = columns[-1]
    df = pd.DataFrame(data=data, columns=columns)
    X_types = {"numerical": ["_a_b.*"], "categorical": ["_a[2-3]_b.*"]}
    prepared = prepare_input_data(
        X=X, y=y, df=df, pos_labels=None, groups=None, X_types=X_types
    )

    df_X, df_y, _, prep_X_types = prepared

    assert all([x in df_X.columns for x in X])
    assert y not in df_X.columns
    assert df_y.name == y
    assert X_types == prep_X_types

    prepared = prepare_input_data(
        X=[":"], y=y, df=df, pos_labels=None, groups=None, X_types=X_types
    )

    df_X, df_y, _, prep_X_types = prepared

    assert all([x in df_X.columns for x in X])
    assert y not in df_X.columns
    assert df_y.name == y
    assert X_types == prep_X_types

    X = columns[:6]
    y = "_a3_b2_c7_"
    prepared = prepare_input_data(
        X=[":"], y=y, df=df, pos_labels=None, groups=None, X_types=X_types
    )

    df_X, df_y, _, prep_X_types = prepared

    assert all([x in df_X.columns for x in X])
    assert y not in df_X.columns
    assert df_y.name == y
    assert X_types == prep_X_types

    X = columns[:6]
    y = "_a3_b2_c7_"
    groups = columns[-1]
    prepared = prepare_input_data(
        X=[":"], y=y, df=df, pos_labels=None, groups=groups, X_types=X_types
    )

    df_X, df_y, df_groups, prep_X_types = prepared

    assert all([x in df_X.columns for x in X])
    assert y not in df_X.columns
    assert groups not in df_X.columns
    assert df_y.name == y
    assert df_groups.name == groups  # type: ignore
    assert X_types == prep_X_types

    X = columns[:6]
    y = "_a3_b2_c7_"
    X_types = {"numerical": ["_a_b.*"]}
    prepared = prepare_input_data(
        X=["_a_.*"], y=y, df=df, pos_labels=None, groups=None, X_types=X_types
    )

    df_X, df_y, _, prep_X_types = prepared

    assert all([x in df_X.columns for x in X])
    assert y not in df_X.columns
    assert df_y.name == y
    assert X_types == prep_X_types

    X = columns[:6]
    y = "_a3_b2_c7_"
    X_types = {"numerical": ["_a_b.*"], "categorical": ["_a[2-3]_b.*"]}
    prepared = prepare_input_data(
        X=[".*_b_.*", ".*a_b2_.*", ".*b3_.*"],
        y=y,
        df=df,
        pos_labels=None,
        groups=None,
        X_types=X_types,
    )

    df_X, df_y, _, prep_X_types = prepared

    assert all([x in df_X.columns for x in X])
    assert y not in df_X.columns
    assert df_y.name == y
    assert X_types == prep_X_types


def test_check_consistency() -> None:
    """Test check_consistency function."""

    # Test binary classification
    y = pd.Series(np.random.randint(0, 2, size=10))
    problem_type = "classification"
    groups = None
    cv = 5

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    # Test multiclass classification
    y = pd.Series(np.random.randint(0, 5, size=10))
    problem_type = "classification"
    groups = None
    cv = 5

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    # Test regression
    y = pd.Series(np.random.randn(10))
    problem_type = "regression"
    groups = None
    cv = 5

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    # Test with wrong problem type
    y = pd.Series(np.random.randint(0, 2, size=10))
    problem_type = "regression"
    groups = None
    cv = 5
    with pytest.warns(RuntimeWarning, match="only 2 distinct values"):
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    y = pd.Series(np.random.rand(10))
    problem_type = "classification"
    groups = None
    cv = 5
    with pytest.warns(RuntimeWarning, match="larger than the number"):
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    # Test with wrong problem type
    y = pd.Series(["A"] * 10)
    problem_type = "regression"
    groups = None
    cv = 5
    with pytest.warns(RuntimeWarning, match="not suitable for a regression"):
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    # Test with only one class
    y = pd.Series(["A"] * 10)
    problem_type = "classification"
    groups = None
    cv = 5
    with pytest.raises(ValueError, match="only one class in y"):
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    # Test CV
    y = pd.Series(np.random.randint(0, 2, size=10))
    problem_type = "classification"
    groups = pd.Series(["A"] * 10)
    cv = 5
    with pytest.warns(
        RuntimeWarning, match="groups was specified but the CV "
    ):
        check_consistency(y=y, cv=cv, groups=groups, problem_type=problem_type)

    valid_instances = (
        GroupKFold(),
        GroupShuffleSplit(),
        LeaveOneGroupOut(),
        LeavePGroupsOut(n_groups=2),
        StratifiedGroupKFold(),
        ContinuousStratifiedGroupKFold(n_bins=2),
        RepeatedContinuousStratifiedGroupKFold(n_bins=2),
    )

    for cv in valid_instances:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_consistency(
                y=y, cv=cv, groups=groups, problem_type=problem_type
            )

    invalid_instances = (
        ShuffleSplit(),
        StratifiedKFold(),
        StratifiedShuffleSplit(),
        LeaveOneOut(),
        LeavePOut(p=2),
    )

    for cv in invalid_instances:
        with pytest.warns(
            RuntimeWarning, match="groups was specified but the CV "
        ):
            check_consistency(
                y=y, cv=cv, groups=groups, problem_type=problem_type
            )


def test__check_x_types() -> None:
    """Test checking for valid X types."""

    X = ["a", "b", "c"]
    X_types = {"categorical": ["a", "b"], "continuous": ["c"]}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checked_X_types = _check_x_types(X=X, X_types=X_types)
        assert X_types == checked_X_types

    X = ["a", "b", "c"]
    X_types = {"categorical": ["a", "b"], "continuous": "c"}
    expected_X_types = {"categorical": ["a", "b"], "continuous": ["c"]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checked_X_types = _check_x_types(X=X, X_types=X_types)
        assert expected_X_types == checked_X_types

    with pytest.warns(
        RuntimeWarning, match="No type checking will be performed"
    ):
        checked_X_types = _check_x_types(X=X, X_types=None)
        assert {} == checked_X_types

    X = ["a", "b", "c"]
    X_types = {"categorical": ["a", "b"]}

    with pytest.warns(RuntimeWarning, match="will be treated as continuous"):
        checked_X_types = _check_x_types(X=X, X_types=X_types)
        assert X_types == checked_X_types

    X = ["a", "b", "c"]
    X_types = {"categorical": ["a", "b", "d"]}
    with pytest.raises(ValueError, match="in X_types but not in X"):
        _check_x_types(X=X, X_types=X_types)

    X = ["a", "b", "c"]
    X_types = {"categorical": ["a", "b"], "continuous": ["a", "c"]}
    with pytest.raises(ValueError, match="more than once in X_types"):
        _check_x_types(X=X, X_types=X_types)

    # Disabling X types check should not raise an error
    set_config("disable_xtypes_check", True)
    X = ["a", "b", "c"]
    X_types = {"categorical": ["a", "b", "d"]}
    _check_x_types(X=X, X_types=X_types)

    X = ["a", "b", "c"]
    X_types = {"categorical": ["a", "b"], "continuous": ["a", "c"]}
    _check_x_types(X=X, X_types=X_types)
    set_config("disable_xtypes_check", False)


def test__check_x_types_regexp() -> None:
    """Test checking for valid X types using regexp."""
    X = [
        "_a_b_c1_",
        "_a_b_c2_",
        "_a_b2_c3_",
        "_a_b2_c4_",
        "_a_b3_c5_",
        "_a_b3_c6_",
        "_a3_b2_c7_",
        "_a2_b_c7_",
        "_a2_b_c8_",
        "_a2_b_c9_",
    ]
    X_types = {"categorical": [".*a_b.*", "_a2.*"], "continuous": ["_a3.*"]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checked_X_types = _check_x_types(X=X, X_types=X_types)
        assert X_types == checked_X_types

    X_types = {
        "categorical": [".*a_b.*", "_a2.*"],
        "continuous": ["_a2_b_c7_"],
    }
    with pytest.raises(ValueError, match="more than once in X_types"):
        _check_x_types(X=X, X_types=X_types)

    X_types = {"categorical": [".*a_b.*"], "continuous": ["_a2_b_c7_"]}
    with pytest.warns(RuntimeWarning, match="not defined in X_types"):
        checked_X_types = _check_x_types(X=X, X_types=X_types)
        assert X_types == checked_X_types

    # Two matching regexp in the same type should not be an issue
    X_types = {
        "categorical": [".*a_b.*", "_a2.*", "_a_b_c.*"],
        "continuous": ["_a3.*"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checked_X_types = _check_x_types(X=X, X_types=X_types)
        assert X_types == checked_X_types
