from julearn.scoring.metrics import r2_corr


def test_r2_corr():
    assert r2_corr([1, 2, 3, 4], [1, 2, 3, 4]) == 1
    assert r2_corr([1, 2, 3, 4], [2, 3, 4, 5]) == 1
