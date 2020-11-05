.. include:: links.inc

Installing JuLearn
==================


Requirements
^^^^^^^^^^^^

Julearn requires the following packages:

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
  latest features. However, it is still under development and not yet officialy
  released. Some features might still change before the next stable release.
* Install from :ref:`install_development_git`. This is mostly suitable for
  developers that want to have the latest version and yet edit the code.


Either way, we strongly recommend using virtual environments:

* `venv`_
* `conda env`_


.. _install_latest_release:

Latest release
--------------

We have packaged our

.. code-block:: bash

    pip install -U julearn


.. _install_latest_development:

Latest Development Version
--------------------------
First, make sure that you have all the dependencies installed:

.. code-block:: bash

    pip install -U scikit-learn pandas


Then, install julearn from TestPypi

.. code-block:: bash

    pip install --index-url https://test.pypi.org/simple/ -U julearn


.. _install_development_git:

Local git repository (for developers)
-------------------------------------
