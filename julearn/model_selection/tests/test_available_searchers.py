# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import pytest
from sklearn.model_selection._search_successive_halving import (
    HalvingGridSearchCV)
from julearn.model_selection import (
    register_searcher, reset_searcher_register, get_searcher)


def test_register_searcher():
    with pytest.raises(ValueError, match='The specified searcher '):
        get_searcher('halving')
    register_searcher('halving', HalvingGridSearchCV)
    assert get_searcher('halving') == HalvingGridSearchCV

    with pytest.warns(RuntimeWarning,
                      match='searcher named halving already exists.'):
        register_searcher('halving', HalvingGridSearchCV)

    register_searcher('halving', HalvingGridSearchCV, overwriting=True)
    with pytest.raises(ValueError,
                       match='searcher named halving already exists and '):

        register_searcher('halving', HalvingGridSearchCV, overwriting=False)

    reset_searcher_register()


def test_reset_searcher():

    register_searcher('halving', HalvingGridSearchCV)
    get_searcher('halving')
    reset_searcher_register()
    with pytest.raises(ValueError,
                       match='The specified searcher '):

        get_searcher('halving')
