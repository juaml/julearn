.. include:: links.inc

.. _contribution_guidelines:

Contributing
============

Setting up the local development environment
--------------------------------------------

#. Fork the https://github.com/juaml/julearn repository on GitHub. If you
   have never done this before, `follow the official guide
   <https://guides.github.com/activities/forking/>`_.
#. Clone your fork locally as described in the same guide.
#. Install your local copy into a Python virtual environment. You can `read
   this guide to learn more
   <https://realpython.com/python-virtual-environments-a-primer/>`_ about them
   and how to create one.

   .. code-block:: bash

       pip install -e ".[dev]"

#. Create a branch for local development using the ``main`` branch as a
   starting point. Use ``fix``, ``refactor``, or ``feat`` as a prefix.

   .. code-block:: bash

       git checkout main
       git checkout -b <prefix>/<name-of-your-branch>

   Now you can make your changes locally.

#. Make sure you install git pre-commit hooks like so:

   .. code-block:: bash

       pre-commit install

#. When making changes locally, it is helpful to ``git commit`` your work
   regularly. On one hand to save your work and on the other hand, the smaller
   the steps, the easier it is to review your work later. Please use `semantic
   commit messages
   <http://karma-runner.github.io/2.0/dev/git-commit-msg.html>`_.

   .. code-block:: bash

       git add .
       git commit -m "<prefix>: <summary of changes>"

   In case, you want to commit some WIP (work-in-progress) code, please indicate
   that in the commit message and use the flag ``--no-verify`` with
   ``git commit`` like so:

   .. code-block:: bash

       git commit --no-verify -m "WIP: <summary of changes>"

#. When you're done making changes, check that your changes pass our test suite.
   This is all included with ``tox``.

   .. code-block:: bash

       tox

   You can also run all ``tox`` tests in parallel. As of ``tox 3.7``, you can run

   .. code-block:: bash

       tox --parallel


#. Push your branch to GitHub.

   .. code-block:: bash

       git push origin <prefix>/<name-of-your-branch>

#. Open the link displayed in the message when pushing your new branch in order
   to submit a pull request. Please follow the template presented to you in the
   web interface to complete your pull request.


GitHub Pull Request guidelines
------------------------------

Before you submit a pull request, check that it meets these guidelines:

#. The pull request should include tests in the respective ``tests`` directory.
   Except in rare circumstances, code coverage must not decrease (as reported
   by codecov which runs automatically when you submit your pull request).
#. If the pull request adds functionality, the docs should be
   updated. Consider creating a Python file that demonstrates the usage in
   ``examples/`` directory.
#. Make sure to create a Draft Pull Request. If you are not sure how to do it,
   check
   `here <https://github.blog/2019-02-14-introducing-draft-pull-requests/>`_.
#. Note the pull request ID assigned after completing the previous step and
   create a short one-liner file of your contribution named as
   ``<pull-request-ID>.<type>`` in ``docs/changes/newsfragments/``, ``<type>``
   being as per the following convention:

   * API change : ``change``
   * Bug fix : ``bugfix``
   * Enhancement : ``enh``
   * Feature : ``feature``
   * Documentation improvement : ``doc``
   * Miscellaneous : ``misc``
   * Deprecation and API removal : ``removal``

   For example, a basic documentation improvement can be recorded in a file
   ``101.doc`` with the content:

   .. code-block::

       Fixed a typo in intro by `julearn's biggest fan`_

#. If it's your first contribution, also add yourself to
   ``docs/changes/contributors.inc``.
#. The pull request will be tested against several Python versions.
#. Someone from the core team will review your work and guide you to a successful
   contribution.


Running unit tests
------------------

julearn uses `pytest <http://docs.pytest.org/en/latest/>`_ for its
unit-tests and new features should in general always come with new
tests that make sure that the code runs as intended.

To run all tests

.. code-block:: bash

    tox -e test


Adding and building documentation
---------------------------------

Building the documentation requires some extra packages and can be installed by

.. code-block:: bash

    pip install -e ".[docs]"

To build the docs

.. code-block:: bash

    cd docs
    make local

To view the documentation, open ``docs/_build/html/index.html``.

In case you remove some files or change their filenames, you can run into
errors when using ``make local``. In this situation you can use ``make clean``
to clean up the already build files and then re-run ``make local``.


Writing Examples
----------------

The format used for text is reST. Check the `sphinx RST reference`_ for more
details. The examples are run and displayed in HTML format using
`sphinx gallery`_. To add an example, just create a ``.py`` file that starts
either with ``plot_`` or ``run_``, depending on whether the example generates
a figure or not.

The first lines of the example should be a Python block comment with a title,
a description of the example, authors and license name.

The following is an example of how to start an example

.. code-block:: python

    """
    Simple Binary Classification
    ============================

    This example uses the 'iris' dataset and performs a simple binary
    classification using a Support Vector Machine classifier.

    """


The rest of the script will be executed as normal Python code. In order to
render the output and embed formatted text within the code, you need to add
a 79 ``#`` (a full line) at the point in which you want to render and add text.
Each line of text shall be preceded with ``#``. The code that is not
commented will be executed.

The following example will create texts and render the output between the
texts.

.. code-block:: python

    from julearn import run_cross_validation
    from julearn.utils import configure_logging
    from seaborn import load_dataset


    ###############################################################################
    # Set the logging level to info to see extra information
    configure_logging(level="INFO")

    ###############################################################################
    # Load the iris dataset
    df_iris = load_dataset("iris")

    ###############################################################################
    # The dataset has three species. We will keep two to perform a binary
    # classification.
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]


Finally, when the example is done, you can run as a normal Python script.
To generate the HTML, just build the docs.
