.. include:: links.inc

Installing julearn
==================


Requirements
^^^^^^^^^^^^

julearn requires the following packages:

* `Python`_ >= 3.6
* `scikit-learn`_
* `pandas`_

Running the examples requires:

* `seaborn`_

Depending on the installation method, this packages might be installed
automatically.

Installing
^^^^^^^^^^
There are different ways to install julearn:

* Install the :ref:`install_latest_release`. This is the most suitable approach
  for most end users.
* Install the :ref:`install_latest_development`. This version will have the
  latest features. However, it is still under development and not yet
  officially released. Some features might still change before the next stable
  release.
* Install from :ref:`install_development_git`. This is mostly suitable for
  developers that want to have the latest version and yet edit the code.


Either way, we strongly recommend using virtual environments:

* `venv`_
* `conda env`_


.. _install_latest_release:

Latest release
--------------

We have packaged julearn and published it in PyPi, so you can just install it
with `pip`.

.. code-block:: bash

    pip install -U julearn


.. _install_latest_development:

Latest Development Version
--------------------------
First, make sure that you have all the dependencies installed:

.. code-block:: bash

    pip install -U scikit-learn pandas numpy

OR:

.. code-block:: bash

    conda install scikit-learn pandas numpy


Then, install julearn from TestPypi

.. code-block:: bash

    pip install --index-url https://test.pypi.org/simple/ -U julearn


.. _install_development_git:

Local git repository (for developers)
-------------------------------------
First, make sure that you have all the dependencies installed:

.. code-block:: bash

    pip install -U scikit-learn pandas

OR:

.. code-block:: bash

    conda install scikit-learn pandas

Then, clone `julearn Github`_ repository in a folder of your choice:

.. code-block:: bash

    git clone https://github.com/juaml/julearn.git


Finally, install in development mode:

.. code-block:: bash

    cd julearn
    python setup.py develop
