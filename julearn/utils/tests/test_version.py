"""Tests for version checks."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from _pytest.monkeypatch import MonkeyPatch

from julearn.utils.versions import _joblib_parallel_args, check_version


def test_major_true() -> None:
    """Test major version check."""
    assert check_version("3.5.1", lambda x: int(x) > 1)


def test_major_false() -> None:
    """Test major version check false."""
    assert check_version("1.5.1", lambda x: int(x) > 1) is False


def test_minor_true() -> None:
    """Test minor version check."""
    assert check_version("3.5.1", minor_check=lambda x: int(x) > 2)


def test_minor_false() -> None:
    """Test minor version check false."""
    assert check_version("3.1.1", minor_check=lambda x: int(x) >= 2) is False


def test_patch_true() -> None:
    """Test patch version check."""
    assert check_version("3.1.5", patch_check=lambda x: int(x) > 2)


def test_patch_false() -> None:
    """Test patch version check false."""
    assert check_version("3.1.1", patch_check=lambda x: int(x) >= 2) is False


def test_multiple_true() -> None:
    """Test multiple checks."""
    assert check_version(
        "3.2.1",
        major_check=lambda x: int(x) == 3,
        minor_check=lambda x: int(x) == 2,
        patch_check=lambda x: int(x) >= 1,
    )


def test_multiple_false() -> None:
    """Test multiple checks false."""
    assert (
        check_version(
            "3.2.1",
            major_check=lambda x: int(x) == 3,
            minor_check=lambda x: int(x) == 3,
            patch_check=lambda x: int(x) >= 2,
        )
        is False
    )


def test_joblib_args_higer_1(monkeypatch: MonkeyPatch) -> None:
    """Test joblib args for sklearn >= 1.0."""
    with monkeypatch.context() as m:
        m.setattr("sklearn.__version__", "2.2.11")
        kwargs = _joblib_parallel_args(prefer="threads")
    assert kwargs["prefer"] == "threads"


def test_joblib_args_lower_1(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test joblib args for sklearn < 1.0."""
    with monkeypatch.context() as m:
        import sklearn

        m.setattr("sklearn.__version__", "0.24.2")
        m.setattr(
            sklearn.utils.fixes,  # type: ignore[attr-defined]
            "_joblib_parallel_args",
            lambda prefer: dict(backend="threads"),
            raising=False,
        )
        kwargs = _joblib_parallel_args(prefer="threads")
    assert kwargs["backend"] == "threads"
