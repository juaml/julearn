.. include:: links.inc

Contributing to julearn
=======================


1. Setting up the development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before your first contribution, you might want to install all the required
tools to be able to test that everything is according to the required
development guidelines.

1.1 Download the source code
****************************

Choose a folder where you will place the code and clone the `julearn Github`_
repository:

.. code-block:: bash

    git clone https://github.com/juaml/julearn.git

1.2 Create a virtual environment
********************************

This step can be ommited, but it is highly recommended, as the development
version of julearn can interfere with the other versions that you might use in
your projects.

Check the `venv`_ or `conda env`_ documentation on how to create one.


1.3 Install the requirements
****************************

The required packages for development are specified within 3 files in the
repository, according to the different build stages.

The following is an example on how to install them using `pip`:

.. code-block:: bash

    cd julearn
    pip install -r requirements.txt
    pip install -r test-requirements.txt
    pip install -r docs-requirements.txt

1.3 Install julearn
*******************

To install julearn in editable/development mode, simple run:

.. code-block:: bash

    python setup.py develop

Now you are ready for the first contribution.

2. Contributing with a pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step, before doing any edit to the code, is to find the issue in the
`julearn Github`_ repository that you will address and explain what your
contribution will do. If it does not exist, you can create it. The idea is that
after your pull request is merged, the issue is fixed/closed.

2.1 Fork the `julearn Github`_ repository
*****************************************
In order to push into a repository, you will need to have push permissions.
Github provides a way of doing this by forking (creating a copy) the repository
into one under your Github user account in.

Simply go to `julearn Github`_  and click on the *Fork* button on the top-right
corner.

Once you do this, there will be a ``julearn`` repository under your username.
For example ``fraimondo/julearn``.

Now add this as an other remote repository to the current julearn local git.
Replace ``fraimondo`` with your Github username.

.. code-block:: bash

    git remote add fraimondo git@github.com:fraimondo/julearn.git


2.2 Create a branch for your modifications
******************************************

All your modifications that address this issue will be placed in a new branch.
Execute the following command, replacing ``<BRANCH_NAME>`` with a name that
relates to the issue.

.. code-block:: bash

    git checkout -b <BRANCH_NAME>


.. _do_changes:

2.3 Do the changes
******************

Simply use your prefered code editor to do the modifications you want.

2.4 Commit and Push
*******************

This is the standard git worflow. Replace ``<USERNAME>>`` with your Github 
username.

    # Add the files you created with ``git add``
    # Commit the changes with ``git commit``
    # Commit the changes with ``git push <USERNAME> <BRANCH_NAME>``

2.5 Test and build the documentation
************************************

Before creating the pull request (PR), make sure that all the test pass and
the documentation can be correctly built.

To run the test, execute:

.. code-block:: bash

    pytest -v

To build the docs:

.. code-block:: bash

    cd doc
    make html

To view the documentation, open `docs/_build/html/index.html`.

If any of this fails, go back to :ref:`do_changes`

2.5 Create a pull request
*************************

Now that you are happy with your contribution, just navigate to your fork
of the julearn repository in `Github`_ and click on the *Compare & pull 
request* button.

Fill in the pull request message and you will be in contact with the julearn
mantainers who will review your contribution. If they suggest any modification,
go back to :ref:`do_changes`.

Once everyone is happy, your modifications will be included in the development
version and later on, in the release version.