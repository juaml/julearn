import pytest
from sklearn.model_selection import ParameterSampler
from julearn.model_selection import (
    register_searcher, reset_searcher_register, get_searcher, list_searchers)


def test_register_searcher():
    with pytest.raises(ValueError, match='The specified searcher '):
        get_searcher('sampler')
    register_searcher('sampler', ParameterSampler)
    assert get_searcher('sampler') == ParameterSampler

    with pytest.warns(RuntimeWarning,
                      match='searcher named sampler already exists.'):
        register_searcher('sampler', ParameterSampler)

    register_searcher('sampler', ParameterSampler, overwriting=True)
    with pytest.raises(ValueError,
                       match='searcher named sampler already exists and '):

        register_searcher('sampler', ParameterSampler, overwriting=False)

    reset_searcher_register()


def test_reset_searcher():

    register_searcher('sampler', ParameterSampler)
    get_searcher('sampler')
    reset_searcher_register()
    with pytest.raises(ValueError,
                       match='The specified searcher '):

        get_searcher('sampler')
