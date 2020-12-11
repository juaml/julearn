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

4. Edit the file ``_version.py`` and remove '.dev'. It should match the next
   release

5. Make sure that test pass and CI won't fail after this modification:

.. code-block:: bash

    pytest -v
    flake8

6. Commit and create tag (replace ``X.Y.Z`` with the proper version).

.. code-block:: bash

    git commit -am "Set version to X.Y.Z"
    git tag vX.Y.Z

7. Check that the build system is creating the proper version

.. code-block:: bash

    SETUPTOOLS_SCM_DEBUG=1 python -m pep517.build --source --binary --out-dir dist/ .

8. Push the tag

.. code-block:: bash

    git push origin vX.Y.Z

9. Edit the file ``_version.py`` with the version of the next release and 
   append '.dev' at the end.

10. Commit and push to main