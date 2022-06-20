import sklearn


def check_version(version,
                  major_check=None, minor_check=None, patch_check=None):
    """check a version following major.minor.patch version numbers
    according to checks as functions major, minor and patch.

    Parameters
    ----------
    version : str

    major_check : func

    minor_check : func

    patch_check : func


    Returns
    -------
    version_checked : bool
    is the version according to the checks




    """

    def get_check(check_func):
        return lambda x: True if check_func is None else check_func(x)

    version_checks = [major_check, minor_check, patch_check]
    versions_checked = [get_check(check)(version)
                        for version, check in zip(
        version.split('.'), version_checks)
    ]

    return all(versions_checked)


def _joblib_parallel_args(**kwargs):
    sklearn_version = sklearn.__version__
    higher_than_11 = check_version(sklearn_version,
                                   lambda x: int(x) >= 1,
                                   lambda x: int(x) >= 1)
    if higher_than_11:
        return kwargs
    else:
        from sklearn.utils.fixes import _joblib_parallel_args as _sk_parallel
        return _sk_parallel(**kwargs)
