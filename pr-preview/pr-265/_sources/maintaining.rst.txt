.. include:: links.inc

Maintaining
===========

Versioning
----------

Version numbers are as follows: *MAJOR.MINOR.MICRO*. Additionally,
development version append *devN* where N is the distance (in commits) to
the last release.

This is done automatically by `setuptools_scm`_.

This plugin reads the latest tagged version from git and automatically
increments the *MICRO* segment and appends *devN*. This is considered a
pre-release.

The CI scripts will publish every tag with the format *v.X.Y.Z* to PyPI as
version "X.Y.Z". Additionally, for every push to main, it will be published
as pre-release to PyPI.

Releasing a new version
-----------------------

Once the milestone is reached (all issues closed), it is time to do a new
release. Make sure you have
`towncrier <https://towncrier.readthedocs.io/en/stable/index.html>`_ installed
before proceeding.

#. Make sure you are in sync with the main branch.

   .. code-block:: bash

       git checkout main
       git pull --rebase origin main

#. Run the following to check changelog is properly generated:

   .. code-block:: bash

      towncrier build --draft

#. Then, run:

   .. code-block:: bash

      towncrier

   to generate the proper changelog that should be reflected in
   ``docs/whats_new.rst``.

#. Commit the changes, make a PR and merge via a merge commit.

#. Make sure you are in sync with the main branch.

   .. code-block:: bash

       git checkout main
       git pull --rebase origin main

#. Create tag (replace ``X.Y.Z`` with the proper version) on the merged PR's
   merge commit.

   .. code-block:: bash

       git tag -a vX.Y.Z -m "Release X.Y.Z"

#. Check that the build system is creating the proper version

   .. code-block:: bash

       SETUPTOOLS_SCM_DEBUG=1 python -m build --outdir dist/ .

#. Push the tag

   .. code-block:: bash

       git push origin --follow-tags

#. Optional: bump the *MAJOR* or *MINOR* segment of next release (replace
   ``D.E.0`` with the proper version).

   .. code-block:: bash

       git tag -a vD.E.0.dev -m "Set next release to D.E.0"
       git push origin --follow-tags
