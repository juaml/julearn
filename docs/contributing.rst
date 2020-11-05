.. include:: links.inc

Contributing to julearn
=======================



Tests
^^^^^

To run the test, just make sure you have the required packages installed (see
`test-requirements.txt`) and run:

.. code-block:: bash

    pytest -v


Documentation
^^^^^^^^^^^^^

To execute the examples and build the documentation, you need to have the
required packages installed (see `doc-requirements.txt`) and run:

.. code-block:: bash

    cd doc
    make html

To force re-building all the docs:

.. code-block:: bash

    cd doc
    make clean
    make html

To view the documentation, open `docs/_build/html/index.html`.