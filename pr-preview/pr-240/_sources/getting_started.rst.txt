.. include:: links.inc


Getting started
===============

Requirements
------------

``julearn`` is compatible with `Python`_ >= 3.8 and requires the following
packages:

* ``numpy>=1.24,<1.26``
* ``pandas>=1.5.0,<2.1``
* ``scikit-learn>=1.2.0``
* ``statsmodels>=0.13,<0.15``

Running the examples require:

* ``seaborn>=0.12.2,<0.13``
* ``bokeh>=3.0.0``
* ``panel>=1.0.0b1``
* ``param>=1.11.0``

Depending on the installation method (e.g. the `pip install` option below),
these packages might be installed automatically. It is nevertheless good to be
aware of these dependencies as installing ``julearn`` might lead to changes in
these packages.

Setup suggestion
================

Although not required, we strongly recommend using **virtual environments** and
installing ``julearn`` into a virtual environment. This helps to keep the setup
clean. The most prominent options are:

* pip: `venv`_
* conda: `conda env`_

Installing
==========

.. note::
    ``julearn`` keeps on being updated and improved. The latest stable release
    and the developer version therefore often differ quite a bit.
    If you want the newest updates, it might make more sense for you to use the
    developer version until we release the next stable ``julearn`` version.


Depending on your aimed usage of ``julearn`` you have two different options
how to install ``julearn``:

#. Install the *latest release*: Likely most suitable for most
   **end users**. This is done by installing the latest stable release from
   PyPI.

   .. code-block:: bash

       pip install -U julearn

#. Install the *latest pre-relase*: This version will have the
   **latest updates**. However, it is still under development and not yet
   officially released. Some features might still change before the next stable
   release.

   .. code-block:: bash

       pip install -U julearn --pre


.. _install_optional_dependencies:

Optional Dependencies
=====================

Some functionality of ``julearn`` requires additional packages. These are not
installed by default. If you want to use these features, you need to specify
them during installation. For example, if you want to use the :mod:`.viz`
module, you need to install the ``viz`` optional dependencies as follows:

.. code-block:: bash

    pip install -U julearn[viz]

The following optional dependencies are available:

* ``viz``: Visualization tools for ``julearn``. This includes the
  :mod:`.viz` module.
* ``deslib``: The :mod:`.dynamic` module requires the `deslib`_ package.
