from julearn.scoring import register_scorer, reset_scorer_register, get_scorer
import pytest


def return_1(estimator, X, y):
    return 1


def test_register_scorer():
    with pytest.raises(ValueError, match='useless is not a valid scorer'):
        get_scorer('useless')
    register_scorer('useless', return_1)
    scorer = get_scorer('useless')

    register_scorer('useless', return_1, True)

    with pytest.warns(RuntimeWarning,
                      match=f'scorer named useless already exists. '):
        register_scorer('useless', return_1, None)

    with pytest.raises(ValueError,
                       match=f'scorer named useless already exists and'):
        register_scorer('useless', return_1, False)
    assert scorer == return_1
    reset_scorer_register()


def test_reset_scorer():
    with pytest.raises(ValueError, match='useless is not a valid scorer '):
        get_scorer('useless')
    register_scorer('useless', return_1)
    get_scorer('useless')
    reset_scorer_register()
    with pytest.raises(ValueError, match='useless is not a valid scorer '):
        get_scorer('useless')
