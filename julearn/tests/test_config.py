"""Provide tests for the config module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

import pytest

from julearn.config import get_config, set_config


def test_set_config_wrong_keys() -> None:
    """Test that set_config raises an error when the key does not exist."""
    with pytest.raises(ValueError, match="does not exist"):
        set_config("wrong_key", 1)


def test_set_get_config() -> None:
    """Test setting and getting config values."""
    old_value = get_config("MAX_X_WARNS")
    new_value = old_value + 1
    set_config("MAX_X_WARNS", new_value)
    assert get_config("MAX_X_WARNS") == new_value
