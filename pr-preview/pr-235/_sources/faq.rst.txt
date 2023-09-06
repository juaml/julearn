.. include:: links.inc

FAQs
====

I have issues with the dependencies for the :mod:`.viz` module.
---------------------------------------------------------------

The :mod:`.viz` module uses `bokeh`_ and `panel`_ to create interactive
plots. These packages are not installed by default when you install
``julearn``. This libraries are also under development and they might not
be as robust as we want.

Usually, installing ``julearn`` with the ``[viz]`` option will install the
necessary dependencies using ``pip``. However, if you have issues with the
installation or you want to install them through other package managers,
you can install them manually.

Using ``pip``:

.. code-block:: bash

  pip install panel
  pip install bokeh

Using ``conda``:

.. code-block:: bash

  conda install -c conda-forge panel
  conda install -c bokeh bokeh


How do I use the :mod:`.viz` interactive plots?
-----------------------------------------------

The interactive plots are based on `bokeh`_ and `panel`_. You can use them
in different ways:

#. As a standalone application, in a browser:

   To do so, you need to call the function ``show`` on the plot object. For
   example:

   .. code-block:: python

       panel = plot_scores(scores1, scores2, scores3)
       panel.show()


#. As part of a Jupyter notebook:

   You will need to install the ``jupyter_bokeh`` package.

   Using ``pip``:

   .. code-block:: bash

       pip install jupyter_bokeh

   Using ``conda``:

   .. code-block:: bash

       conda install -c bokeh jupyter_bokeh

   This will allow you to see the plots interactively in the notebook. To do so,
   you need to call the function ``servable`` on the plot object. For example:

   .. code-block:: python

       panel = plot_scores(scores1, scores2, scores3)
       panel.servable()

.. TODO: As part of a Binder notebook to share with colleagues.
