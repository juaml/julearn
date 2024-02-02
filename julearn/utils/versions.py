"""Utils for handling scikit-learn versions."""

from typing import Any, Callable, Optional

import sklearn


def check_version(
    version: str,
    major_check: Optional[Callable] = None,
    minor_check: Optional[Callable] = None,
    patch_check: Optional[Callable] = None,
):
    """Check a version following major.minor.patch version numbers.

    The version is checked according to checks as functions major, minor and
    patch. This functions must take a string and return a boolean.

    Parameters
    ----------
    version : str
        version to check

    major_check : Callable
        function to check major version

    minor_check : func
        function to check minor version

    patch_check : func
        function to check patch version

    Returns
    -------
    version_checked : bool
        if the version passes the checks

    """

    def get_check(check_func):
        return lambda x: True if check_func is None else check_func(x)

    version_checks = [major_check, minor_check, patch_check]
    versions_checked = [
        get_check(check)(version)
        for version, check in zip(version.split("."), version_checks)
    ]

    return all(versions_checked)


def _joblib_parallel_args(**kwargs: Any) -> Any:
    """Get joblib parallel args depending on scikit-learn version.

    Parameters
    ----------
    **kwargs : dict
        keyword arguments to pass to joblib.Parallel

    """
    sklearn_version = sklearn.__version__
    higher_than_11 = check_version(
        sklearn_version, lambda x: int(x) >= 1, lambda x: int(x) >= 1
    )
    if higher_than_11:
        return kwargs
    else:
        from sklearn.utils.fixes import (
            _joblib_parallel_args as _sk_parallel,  # type: ignore
        )

        return _sk_parallel(**kwargs)
