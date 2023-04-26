.. include:: links.inc

*************************
Frequenly Asked Questions
*************************

1. How do I use the julearn :mod:`.viz` interactive plots?
----------------------------------------------------------

The interactive plots are based on `bokeh`_ and `panel`_. You can use them
in different ways:

1. As a standalone application, in a browser.

To do so, you need to call the function ``show`` on the plot object. For
example:

.. code-block:: python

    panel = plot_scores(scores1, scores2, scores3)
    panel.show()


2. As part of a Jupyter notebook.

You will need to install the ``jupyter_bokeh`` package.

Using conda:

.. code-block:: bash

    conda install -c bokeh jupyter_bokeh

Using pip:

.. code-block:: bash

    pip install jupyter_bokeh

This will allow you to see the plots interactively in the notebook. To do so,
you need to call the function ``servable`` on the plot object. For example:

.. code-block:: python

    panel = plot_scores(scores1, scores2, scores3)
    panel.servable()

.. TODO: As part of a Binder notebook to share with colleagues.
