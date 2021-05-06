# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import logging
import subprocess
import sys
from distutils.version import LooseVersion
from pathlib import Path
import warnings

logger = logging.getLogger('julearn')


def _get_git_head(path):
    """Aux function to read HEAD from git"""
    if not path.exists():
        raise_error('This path does not exist: {}'.format(path))
    command = ('cd {gitpath}; '
               'git rev-parse --verify HEAD').format(gitpath=path)
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               shell=True)
    proc_stdout = process.communicate()[0].strip()
    del process
    return proc_stdout


def get_versions(sys):
    """Get versions for each module. If it's a git-installed package, get
    the git hash too.

    Parameters
    ----------
    sys : module
        The sys module object.

    Returns
    -------
    module_versions : dict
        The module names and corresponding versions.
    """
    module_versions = {}
    for name, module in sys.modules.items():
        if '.' in name:
            continue
        if name in ['_curses', '_glmnet']:
            continue
        module_version = LooseVersion(getattr(module, '__version__', None))
        module_version = getattr(module_version, 'vstring', None)
        if module_version is None:
            module_version = None
        elif 'git' in module_version:
            git_path = Path(module.__file__).resolve().parent
            head = _get_git_head(git_path)
            module_version += '-HEAD:{}'.format(head)

        module_versions[name] = module_version
    return module_versions


def _safe_log(versions, name):
    if name in versions:
        logger.info(f'{name}: {versions[name]}')


def log_versions():
    """Log versions of the core libraries, for reproducibility purposes."""
    versions = get_versions(sys)
    logger.info('===== Lib Versions =====')
    _safe_log(versions, 'numpy')
    _safe_log(versions, 'scipy')
    _safe_log(versions, 'sklearn')
    _safe_log(versions, 'pandas')
    _safe_log(versions, 'julearn')

    logger.info('========================')


_logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                      WARNING=logging.WARNING, ERROR=logging.ERROR)


def configure_logging(level='WARNING', fname=None, overwrite=None,
                      output_format=None):
    """Configure the logging functionality

    Parameters
    ----------
    level : int or string
        The level of the messages to print. If string, it will be interpreted
        as elements of logging.
        Options are: ['DEBUG', 'INFO', 'WARNING', 'ERROR']. Defaults to
        'WARNING'.
    fname : str, Path or None
        Filename of the log to print to. If None, stdout is used.
    overwrite : bool | None
        Overwrite the log file (if it exists). Otherwise, statements
        will be appended to the log (default). None is the same as False,
        but additionally raises a warning to notify the user that log
        entries will be appended.
    output_format : str
        Format of the output messages. See the following for examples:

            https://docs.python.org/dev/howto/logging.html

        e.g., "%(asctime)s - %(levelname)s - %(message)s".

        Defaults to "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """
    _close_handlers(logger)
    if output_format is None:
        output_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(output_format)

    if fname is not None:
        if not isinstance(fname, Path):
            fname = Path(fname)
        if fname.exists() and overwrite is None:
            warnings.warn(
                f'File ({fname.as_posix()}) exists. '
                'Messages will be appended. Use overwrite=True to '
                'overwrite or overwrite=False to avoid this message')
            overwrite = False
        mode = 'w' if overwrite else 'a'
        lh = logging.FileHandler(fname, mode=mode)
    else:
        lh = logging.StreamHandler(WrapStdOut())

    if isinstance(level, str):
        level = _logging_types[level]
    lh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(lh)
    log_versions()


def _close_handlers(logger):
    for handler in list(logger.handlers):
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger.removeHandler(handler)


def raise_error(msg, klass=ValueError):
    """Raise an error, but first log it

    Parameters
    ----------
    msg : str
        Error message
    klass : class of the error to raise. Defaults to ValueError
    """
    logger.error(msg)
    raise klass(msg)


def warn(msg, category=RuntimeWarning):
    """Warn, but first log it

    Parameters
    ----------
    msg : str
        Warning message
    category : instance of Warning
        The warning class. Defaults to ``RuntimeWarning``.
    """
    logger.warning(msg)
    warnings.warn(msg, category=category)


class WrapStdOut(object):
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.
    """

    def __getattr__(self, name):  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError(f"'file' object has not attribute '{name}'")
