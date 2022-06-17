from julearn.utils.versions import check_version, _joblib_parallel_args


def test_major_true():
    assert check_version("3.5.1", lambda x: int(x) > 1)


def test_major_false():
    assert check_version("1.5.1", lambda x: int(x) > 1) is False


def test_minor_true():
    assert check_version("3.5.1", minor_check=lambda x: int(x) > 2)


def test_minor_false():
    assert check_version("3.1.1", minor_check=lambda x: int(x) >= 2) is False


def test_patch_true():
    assert check_version("3.1.5", patch_check=lambda x: int(x) > 2)


def test_patch_false():
    assert check_version("3.1.1", patch_check=lambda x: int(x) >= 2) is False


def test_multiple_true():
    assert check_version("3.2.1",
                         major_check=lambda x: int(x) == 3,
                         minor_check=lambda x: int(x) == 2,
                         patch_check=lambda x: int(x) >= 1)


def test_multiple_false():
    assert check_version("3.2.1",
                         major_check=lambda x: int(x) == 3,
                         minor_check=lambda x: int(x) == 3,
                         patch_check=lambda x: int(x) >= 2) is False


def test_joblib_args_higer_1(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr("sklearn.__version__", "2.2.11")
        kwargs = _joblib_parallel_args(prefer="threads")
    assert kwargs["prefer"] == "threads"


def test_joblib_args_lower_1(monkeypatch):
    with monkeypatch.context() as m:
        import sklearn
        m.setattr("sklearn.__version__", "0.24.2")
        m.setattr(
            sklearn.utils.fixes, "_joblib_parallel_args",
            lambda prefer: dict(backend="threads"), raising=False
        )
        kwargs = _joblib_parallel_args(prefer="threads")
    assert kwargs["backend"] == "threads"
