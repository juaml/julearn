.. include:: links.inc

Contributing to julearn
=======================


Setting up the development environment
--------------------------------------

Before your first contribution, you might want to install all the required
tools to be able to test that everything is according to the required
development guidelines.

Download the source code
^^^^^^^^^^^^^^^^^^^^^^^^

Choose a folder where you will place the code and clone the `julearn Github`_
repository:

.. code-block:: bash

    git clone https://github.com/juaml/julearn.git

Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step can be omitted, but it is highly recommended, as the development
version of julearn can interfere with the other versions that you might use in
your projects.

Check the `venv`_ or `conda env`_ documentation on how to create one.


Install the requirements
^^^^^^^^^^^^^^^^^^^^^^^^

The required packages for development are specified within 3 files in the
repository, according to the different build stages.

The following is an example on how to install them using `pip`:

.. code-block:: bash

    cd julearn
    pip install -r requirements.txt
    pip install -r test-requirements.txt
    pip install -r docs-requirements.txt
    pip install -r dev-requirements.txt

Install julearn
^^^^^^^^^^^^^^^

To install julearn in editable/development mode, simple run:

.. code-block:: bash

    python setup.py develop

Now you are ready for the first contribution.

.. note:: Every time that you run ``setup.py develop``, the version is going to
  be automatically set based on the git history. Nevertheless, this change 
  should not be committed (changes to ``_version.py``). Running ``git stash``
  at this point will forget the local changes to ``_version.py``.


Contributing with a pull request
--------------------------------

The first step, before doing any edit to the code, is to find the issue in the
`julearn Github`_ repository that you will address and explain what your
contribution will do. If it does not exist, you can create it. The idea is that
after your pull request is merged, the issue is fixed/closed.

Fork the `julearn Github`_ repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to push into a repository, you will need to have push permissions.
Github provides a way of doing this by forking (creating a copy) the repository
into one under your Github user account in.

Simply go to `julearn Github`_  and click on the ^Fork^ button on the top-right
corner.

Once you do this, there will be a ``julearn`` repository under your username.
For example ``fraimondo/julearn``.

Now add this as an other remote repository to the current julearn local git.
Replace ``fraimondo`` with your Github username.

.. code-block:: bash

    git remote add fraimondo git@github.com:fraimondo/julearn.git


Create a branch for your modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All your modifications that address this issue will be placed in a new branch.
First, make sure you have the latest commit in your local ``main`` branch.

.. code-block:: bash

    git checkout main
    git pull --rebase origin main


Then, execute the following command to create a new branch, replacing 
``<BRANCH_NAME>`` with a name that relates to the issue.

.. code-block:: bash

    git checkout -b <BRANCH_NAME>


.. _do_changes:

Do the changes
^^^^^^^^^^^^^^

Simply use your preferred code editor to do the modifications you want.

.. note:: You must also add a line in `docs/changes/latest.inc` where you briefly
  explain the modifications you made. If it's your first contribution, also
  add yourself to `docs/changes/contributors.inc`.


Commit and Push
^^^^^^^^^^^^^^^

This is the standard git workflow. Replace ``<USERNAME>>`` with your Github 
username.

    # Add the files you created with ``git add``
    # Commit the changes with ``git commit``
    # Commit the changes with ``git push <USERNAME> <BRANCH_NAME>``

Test and build the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before creating the pull request (PR), make sure that all the test pass and
the documentation can be correctly built.

To check the code style, run:

.. code-block:: bash

    flake8

To run the spell check, run:

.. code-block:: bash

    codespell julearn/ docs/ examples/

To run the test, execute:

.. code-block:: bash

    pytest -v


To build the docs:

.. code-block:: bash

    cd docs
    make html

To view the documentation, open `docs/_build/html/index.html`.

In case you remove some files or change their filename you can run into
errors when using ``make html``. In this situation you can use ``make clean``
to clean up the already build files and then rerun ``make html``.

If any of this fails, go back to :ref:`do_changes`

Create a pull request
^^^^^^^^^^^^^^^^^^^^^

Now that you are happy with your contribution, just navigate to your fork
of the julearn repository in `Github`_ and click on the ^Compare & pull 
request^ button.

Fill in the pull request message and you will be in contact with the julearn
manitainers who will review your contribution. If they suggest any
modification, go back to :ref:`do_changes`.

Once everyone is happy, your modifications will be included in the development
version and later on, in the release version.



Writing Examples
----------------

Examples are run and displayed in HTML format using `sphinx gallery`_. There
are two sub directories in the ``examples`` directory: ``basic`` and
``advanced``. To add an example, just create a ``.py`` file that starts either
with ``plot_`` or ``run_``, dependending on whether the example generates a 
figure or not.

The first lines of the example should be a python block comment with a title,
a description of the example an the following include directive to be able to
use the links.

The format use for text is RST. Check the `sphinx RST reference`_ for more
details.

Example of the first lines:


.. code-block:: python

    """
    Simple Binary Classification
    ============================

    This example uses the 'iris' dataset and performs a simple binary
    classification using a Support Vector Machine classifier.

    .. include:: ../../links.inc
    """


The rest of the script will be executed as normal python code. In order to
render the output and embed formatted text within the code, you need to add
a 79 ``#`` (a full line) at the point in which you want to render and add text.
Each line of text shall be preceded with ``#``. The code that it's not
commented will be executed.

The following example will create 3 texts and render the output between the
texts.

.. code-block:: python

    ###############################################################################
    # Imports needed for the example
    from seaborn import load_dataset
    from julearn import run_cross_validation
    from julearn.utils import configure_logging

    ###############################################################################
    df_iris = load_dataset('iris')

    ###############################################################################
    # The dataset has three kind of species. We will keep two to perform a binary
    # classification.
    df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]



Finally, when the example is done, you can run as a normal python script.
To generate the HTML, just build the docs:

.. code-block:: bash

    cd docs
    make html

