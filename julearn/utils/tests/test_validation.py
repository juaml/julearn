import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from julearn.utils.validation import check_n_confounds, is_transformable


def test_check_n_confounds():
    with pytest.raises(ValueError,
                       match='n_confounds has to be an int'):
        check_n_confounds('1')

    with pytest.raises(ValueError,
                       match='n_confounds needs to be >=0'):
        check_n_confounds(-1)
    check_n_confounds(1)
    check_n_confounds(0)


def test_is_transformable():

    assert is_transformable(StandardScaler())
    assert not is_transformable(LogisticRegression)
