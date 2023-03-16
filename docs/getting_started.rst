.. include:: links.inc

Getting started
================


Requirements
^^^^^^^^^^^^

Julearn requires the following packages:

* `Python`_ >= 3.8
* `pandas`_ >= 1.4.0, < 1.6
* `scikit-learn`_ == 1.2.0rc1

Running the examples requires:

* `seaborn`_ >= 0.11.2, < 0.12

Depending on the installation method (e.g. the `pip install`-option below),
these packages might be installed automatically. It is nevertheless good to be
aware of these dependencies as installing Julearn might lead to changes in these 
packages.

Setup suggestion
^^^^^^^^^^^^^^^^^
Although not required, we strongly recommend using **virtual environments** and 
installing Julearn into a virtual environment. This helps to keep the setup
clean. The most prominent options are:

* pip: `venv`_
* conda: `conda env`_

Installing
^^^^^^^^^^

.. note::
    Julearn keeps on being updated and improved. The latest stable release and
    the developer version therefore oftentimes differ quite a bit.
    If you want the newest updates it might make more sense for you to use the
    developer version until we release the next stable julearn version.


Depending on your aimed usage of Julearn you have three different options 
how to install Julearn:

1. Install the :ref:`install_latest_release`: Likely most suitable for most **end 
   users**. However, missing the latest features.
2. Install the :ref:`install_latest_development`: This version will have the
   **latest features**. However, it is still under development and not yet
   officially released. Some features might still change before the next stable
   release.
3. Install from :ref:`install_development_git`: This is mostly suitable for
   **developers** that want to have the latest version and yet edit the code.

.. _install_latest_release:

Latest Stable Release
---------------------

The latest stable release of Julearn is packaged and published in PyPi. 
You can install it (optimally into your virtual environment!) using `pip`:

.. code-block:: bash

    pip install -U julearn


.. _install_latest_development:

Latest Development Version
--------------------------
1. Make sure that you have all the dependencies installed:
   
   **Either** use `pip`:

   .. code-block:: bash

    pip install -U scikit-learn pandas

   **OR** use `conda`:
   
   .. code-block:: bash

    conda install scikit-learn pandas

2. Install julearn from TestPypi:
   
   .. code-block:: bash

    pip install --index-url https://test.pypi.org/simple/ -U julearn --pre


.. _install_development_git:

Local git repository (for developers)
-------------------------------------
1. Make sure that you have all the dependencies installed:
   
   **Either** use `pip`:

   .. code-block:: bash

    pip install -U scikit-learn pandas

   **OR** use `conda`:
   
   .. code-block:: bash

    conda install scikit-learn pandas

2. Clone the `julearn Github`_ repository in a folder of your choice:
   
   .. code-block:: bash

    git clone https://github.com/juaml/julearn.git

3. Switch into the Julearn directory and install the development-mode 
   requirements:
   
   .. code-block:: bash

    cd julearn
    pip install -r dev-requirements.txt

4. Install in development-mode:
   
   .. code-block:: bash

    python setup.py develop
    
   .. note:: Every time that you run ``setup.py develop``, the version is going to 
    be automatically set based on the git history. Nevertheless, this change 
    (changes to ``_version.py``) should not be committed. Running ``git stash``
    at this point to make forget the local changes to ``_version.py``.
