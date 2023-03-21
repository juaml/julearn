"""Test JuTargetTransformer class."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pytest

from julearn.transformers.target import JuTargetTransformer


def test_JuTargetTransformer_abstractness() -> None:
    """Test JuTargetTransformer is abstract base class."""
    with pytest.raises(NotImplementedError, match=r"fit"):
        JuTargetTransformer().fit("1", "2")  # type: ignore
