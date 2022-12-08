from julearn.scoring.metrics import r2_corr, r_pearson


def test_r2_corr():
    """Test r2_corr."""
    assert r2_corr([1, 2, 3, 4], [1, 2, 3, 4]) == 1
    assert r2_corr([1, 2, 3, 4], [2, 3, 4, 5]) == 1


def test_r_pearson():
    """Test r_pearson."""
    assert r_pearson([0, 1, 2, 4, 5, 6], [0, 1, 2, 4, 5, 6]) == 1
    assert r_pearson([0, 1, 2, 4, 5, 6], [6, 5, 4, 2, 1, 0]) == -1
    assert r_pearson([0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]) == 1
