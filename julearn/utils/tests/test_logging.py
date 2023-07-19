"""Test for the logging module."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import tempfile
from pathlib import Path

import pytest

from julearn.utils import configure_logging, logger, raise_error, warn_with_log
from julearn.utils.logging import _close_handlers


# TODO: Fix this deprecation warning issue by fixing the real problem
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_log_file() -> None:
    """Test logging to a file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        configure_logging(fname=tmpdir / "test1.log")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warn message")
        logger.error("Error message")
        _close_handlers(logger)
        with open(tmpdir / "test1.log") as f:
            lines = f.readlines()
            assert not any("Debug message" in line for line in lines)
            assert not any("Info message" in line for line in lines)
            assert any("Warn message" in line for line in lines)
            assert any("Error message" in line for line in lines)

        configure_logging(fname=tmpdir / "test2.log", level="INFO")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warn message")
        logger.error("Error message")
        _close_handlers(logger)
        with open(tmpdir / "test2.log") as f:
            lines = f.readlines()
            assert not any("Debug message" in line for line in lines)
            assert any("Info message" in line for line in lines)
            assert any("Warn message" in line for line in lines)
            assert any("Error message" in line for line in lines)

        configure_logging(fname=tmpdir / "test3.log", level="WARNING")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warn message")
        logger.error("Error message")
        _close_handlers(logger)
        with open(tmpdir / "test3.log") as f:
            lines = f.readlines()
            assert not any("Debug message" in line for line in lines)
            assert not any("Info message" in line for line in lines)
            assert any("Warn message" in line for line in lines)
            assert any("Error message" in line for line in lines)

        configure_logging(fname=tmpdir / "test4.log", level="ERROR")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warn message")
        logger.error("Error message")
        with open(tmpdir / "test4.log") as f:
            lines = f.readlines()
            assert not any("Debug message" in line for line in lines)
            assert not any("Info message" in line for line in lines)
            assert not any("Warn message" in line for line in lines)
            assert any("Error message" in line for line in lines)

        with pytest.warns(UserWarning, match="to avoid this message"):
            configure_logging(fname=tmpdir / "test4.log", level="WARNING")
            logger.debug("Debug2 message")
            logger.info("Info2 message")
            logger.warning("Warn2 message")
            logger.error("Error2 message")
            with open(tmpdir / "test4.log") as f:
                lines = f.readlines()
                assert not any("Debug message" in line for line in lines)
                assert not any("Info message" in line for line in lines)
                assert not any("Warn message" in line for line in lines)
                assert any("Error message" in line for line in lines)
                assert not any("Debug2 message" in line for line in lines)
                assert not any("Info2 message" in line for line in lines)
                assert any("Warn2 message" in line for line in lines)
                assert any("Error2 message" in line for line in lines)

        configure_logging(
            fname=tmpdir / "test4.log", level="WARNING", overwrite=True
        )
        logger.debug("Debug3 message")
        logger.info("Info3 message")
        logger.warning("Warn3 message")
        logger.error("Error3 message")
        with open(tmpdir / "test4.log") as f:
            lines = f.readlines()
            assert not any("Debug message" in line for line in lines)
            assert not any("Info message" in line for line in lines)
            assert not any("Warn message" in line for line in lines)
            assert not any("Error message" in line for line in lines)
            assert not any("Debug2 message" in line for line in lines)
            assert not any("Info2 message" in line for line in lines)
            assert not any("Warn2 message" in line for line in lines)
            assert not any("Error2 message" in line for line in lines)
            assert not any("Debug3 message" in line for line in lines)
            assert not any("Info3 message" in line for line in lines)
            assert any("Warn3 message" in line for line in lines)
            assert any("Error3 message" in line for line in lines)

        with pytest.warns(RuntimeWarning, match=r"Warn raised"):
            warn_with_log("Warn raised")
        with pytest.raises(ValueError, match=r"Error raised"):
            raise_error("Error raised")
        with open(tmpdir / "test4.log") as f:
            lines = f.readlines()
            assert any("Warn raised" in line for line in lines)
            assert any("Error raised" in line for line in lines)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_log() -> None:
    """Simple log test."""
    configure_logging()
    logger.info("Testing")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_lib_logging() -> None:
    """Test logging versions."""

    import numpy as np  # noqa
    import pandas  # noqa
    import scipy  # noqa
    import sklearn  # noqa

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        configure_logging(fname=tmpdir / "test1.log", level="INFO")
        logger.info("first message")
        with open(tmpdir / "test1.log") as f:
            lines = f.readlines()
            assert any("numpy" in line for line in lines)
            assert any("scipy" in line for line in lines)
            assert any("sklearn" in line for line in lines)
            assert any("pandas" in line for line in lines)
            assert any("julearn" in line for line in lines)
