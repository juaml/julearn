.. include:: links.inc

Maintaining julearn
===================


Versioning
^^^^^^^^^^
julearn version numbers are as follows: *MAJOR.MINOR.MICRO*. Additionally,
development version append *devN* where N is the distance (in commits) to
the last release.

This is done automatically by `setuptools_scm`_.

This plugin reads the latest tagged version from git and automatically
increments the *MICRO* segment and appends *devN*. This is considered a
pre-release.

The CI scripts will publish every tag with the format *v.X.Y.Z* to Pypi as
version "X.Y.Z". Additionally, for every push to main, it will be published
as pre-release to TestPypi.

Releasing a new version
^^^^^^^^^^^^^^^^^^^^^^^
Once the milestone is reached (all issues closed), it is time to do a new
release.

1. Open a PR to update documentation:
    - `docs/changes/latest.inc` should now be renamed to the version to be
      released.
    - `docs/whats_new.rst` must update the link to the new version.

2. Merge PR

3. Make sure you are in sync with the main branch.

.. code-block:: bash

    git checkout main
    git pull --rebase origin main

4. Create tag (replace ``X.Y.Z`` with the proper version).

.. code-block:: bash

    git tag vX.Y.Z

5. Check that the build system is creating the proper version

.. code-block:: bash

    SETUPTOOLS_SCM_DEBUG=1 python -m build --source --binary --out-dir dist/ .

6. Push the tag

.. code-block:: bash

    git push origin vX.Y.Z