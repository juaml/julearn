"""Logging utilities for julearn."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import logging
import sys
import warnings
from distutils.version import LooseVersion
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Dict, NoReturn, Optional, Type, Union


logger = logging.getLogger("julearn")


def _get_git_head(path: Path) -> str:
    """Aux function to read HEAD from git.

    Parameters
    ----------
    path : pathlib.Path
        The path to read git HEAD from.

    Returns
    -------
    str
        Empty string if timeout expired for subprocess command execution else
        git HEAD information.

    """
    if not path.exists():
        raise ValueError(f"This path does not exist: {path}")
    command = f"cd {path}; git rev-parse --verify HEAD"
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True,
    )
    try:
        stdout, _ = process.communicate(timeout=10)
        proc_stdout = stdout.strip().decode()
    except TimeoutExpired:
        process.kill()
        proc_stdout = ""
    return proc_stdout


def get_versions() -> Dict:
    """Import stuff and get versions if module.

    Returns
    -------
    module_versions : dict
        The module names and corresponding versions.

    """
    module_versions = {}
    for name, module in sys.modules.items():
        if "." in name:
            continue
        if name in ["_curses"]:
            continue
        vstring = str(getattr(module, "__version__", None))
        module_version = LooseVersion(vstring)
        module_version = getattr(module_version, "vstring", None)
        if module_version is None:
            module_version = None
        elif "git" in module_version:
            git_path = Path(module.__file__).resolve().parent  # type: ignore
            head = _get_git_head(git_path)
            module_version += f"-HEAD:{head}"

        module_versions[name] = module_version
    return module_versions


def _safe_log(versions: Dict, name: str) -> None:
    """Log with safety.

    Parameters
    ----------
    versions : dict
        The dictionary with keys as dependency names and values as the
        versions.
    name : str
        The dependency to look up in `versions`.

    """
    if name in versions:
        logger.info(f"{name}: {versions[name]}")


def log_versions() -> None:
    """Log versions of the core libraries, for reproducibility purposes."""
    versions = get_versions()
    logger.info("===== Lib Versions =====")
    _safe_log(versions, "numpy")
    _safe_log(versions, "scipy")
    _safe_log(versions, "sklearn")
    _safe_log(versions, "pandas")
    _safe_log(versions, "julearn")

    logger.info("========================")


_logging_types = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def configure_logging(
    level: Union[int, str] = "WARNING",
    fname: Optional[Union[str, Path]] = None,
    overwrite: Optional[bool] = None,
    output_format=None,
) -> None:
    """Configure the logging functionality.

    Parameters
    ----------
    level : int or {"DEBUG", "INFO", "WARNING", "ERROR"}
        The level of the messages to print. If string, it will be interpreted
        as elements of logging (default "WARNING").
    fname : str or pathlib.Path, optional
        Filename of the log to print to. If None, stdout is used
        (default None).
    overwrite : bool, optional
        Overwrite the log file (if it exists). Otherwise, statements
        will be appended to the log (default). None is the same as False,
        but additionally raises a warning to notify the user that log
        entries will be appended (default None).
    output_format : str, optional
        Format of the output messages. See the following for examples:
        https://docs.python.org/dev/howto/logging.html
        e.g., ``"%(asctime)s - %(levelname)s - %(message)s"``.
        If None, default string format is used
        (default ``"%(asctime)s - %(name)s - %(levelname)s - %(message)s"``).

    """
    _close_handlers(logger)  # close relevant logger handlers

    # Set logging level
    if isinstance(level, str):
        level = _logging_types[level]

    # Set logging output handler
    if fname is not None:
        # Convert str to Path
        if not isinstance(fname, Path):
            fname = Path(fname)
        if fname.exists() and overwrite is None:
            warnings.warn(
                f"File ({fname.absolute()!s}) exists. "
                "Messages will be appended. Use overwrite=True to "
                "overwrite or overwrite=False to avoid this message.",
                stacklevel=2,
            )
            overwrite = False
        mode = "w" if overwrite else "a"
        lh = logging.FileHandler(fname, mode=mode)
    else:
        lh = logging.StreamHandler(WrapStdOut())  # type: ignore

    # Set logging format
    if output_format is None:
        output_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # (
        #     "%(asctime)s [%(levelname)s] %(message)s "
        #     "(%(filename)s:%(lineno)s)"
        # )
    formatter = logging.Formatter(fmt=output_format)

    lh.setFormatter(formatter)  # set formatter
    logger.setLevel(level)  # set level
    logger.addHandler(lh)  # set handler
    log_versions()  # log versions of installed packages


def _close_handlers(logger: logging.Logger) -> None:
    """Safely close relevant handlers for logger.

    Parameters
    ----------
    logger : logging.logger
        The logger to close handlers for.

    """
    for handler in list(logger.handlers):
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger.removeHandler(handler)


def raise_error(
    msg: str,
    klass: Type[Exception] = ValueError,
    exception: Optional[Exception] = None,
) -> NoReturn:
    """Raise error, but first log it.

    Parameters
    ----------
    msg : str
        The message for the exception.
    klass : subclass of Exception, optional
        The subclass of Exception to raise using (default ValueError).
    exception : Exception, optional
        The original exception to follow up on (default None).

    """
    logger.error(msg)
    if exception is not None:
        raise klass(msg) from exception
    else:
        raise klass(msg)


def warn_with_log(
    msg: str, category: Optional[Type[Warning]] = RuntimeWarning
) -> None:
    """Warn, but first log it.

    Parameters
    ----------
    msg : str
        Warning message.
    category : subclass of Warning, optional
        The warning subclass (default RuntimeWarning).

    """

    # This is somehow nasty. If there is a filter active, then the warning
    # will still be logged. So we need to check if any of the filters
    # will ignore this warning. If this is the case, then do not log it.
    this_filters = [x for x in warnings.filters if issubclass(x[2], category)]
    skip_log = len(this_filters) > 0 and this_filters[0][0] == "ignore"
    if not skip_log:
        logger.warning(msg)
    warnings.warn(msg, category=category, stacklevel=2)


class WrapStdOut(logging.StreamHandler):
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.

    """

    def __getattr__(self, name: str) -> str:
        """Implement attribute fetch."""
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError(f"'file' object has not attribute '{name}'")
