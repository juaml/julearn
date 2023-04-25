.. include:: links.inc

###############
Getting started
###############


Requirements
============

Julearn requires the following packages:

* `Python`_ >= 3.8
* `pandas`_ >= 1.4.0, < 1.6
* `scikit-learn`_ == 1.2.0rc1

Running the examples requires:

* `seaborn`_ >= 0.11.2, < 0.12
* `bokeh`_ >= 3.0.2
* `panel`_ >= 1.0.0b1
* `param`_ >= 1.12.0

Depending on the installation method (e.g. the `pip install` option below),
these packages might be installed automatically. It is nevertheless good to be
aware of these dependencies as installing Julearn might lead to changes in
these packages.

Setup suggestion
================

Although not required, we strongly recommend using **virtual environments** and
installing Julearn into a virtual environment. This helps to keep the setup
clean. The most prominent options are:

* pip: `venv`_
* conda: `conda env`_

Installing
==========

.. note::
    Julearn keeps on being updated and improved. The latest stable release and
    the developer version therefore oftentimes differ quite a bit.
    If you want the newest updates it might make more sense for you to use the
    developer version until we release the next stable julearn version.


Depending on your aimed usage of Julearn you have two different options
how to install Julearn:

1. Install the *latest release*: Likely most suitable for most
   **end users**. This is done by installing the latest stable release from
   PyPi.

.. code-block:: bash

    pip install -U julearn

2. Install the *latest pre-relase*: This version will have the
   **latest updates**. However, it is still under development and not yet
   officially released. Some features might still change before the next stable
   release.

.. code-block:: bash

    pip install -U julearn --pre


.. _install_optional_dependencies:

Optional Dependencies
=====================

Some functionality of Julearn requires additional packages. These are not
insalled by default. If you want to use these features, you need to specify
them during installation. For example, if you want to use the `:mod:.viz`
module, you need to install the ``viz`` optional dependencies as follows:

.. code-block:: bash

    pip install -U julearn[viz]

The following optional dependencies are available:

* ``viz``: Visualization tools for Julearn. This includes the
  `:mod:.viz` module.
* ``deslib``: The :mod:`.dynamic` module requires the `deslib`_ package.
