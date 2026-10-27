.. include:: ../links.inc

.. _need_to_know:

What you really need to know
============================

The backbone of ``julearn`` is the function :func:`.run_cross_validation`, which
allows you to do all the *magic*. All important information needed to estimate
your machine learning workflow's performance goes into this function, specified
via its parameters.

But why is basically everything based on one *cross-validation* function? Well,
because doing proper cross-validation is of utmost importance in machine
learning and it is not as easy as it might seem at first glance. If you want to
understand why, reading the sub-chapter :ref:`cross_validation` is a good
starting point.

Once you are familiar with the basics of *cross-validation*, you can follow
along the other sub-chapters to learn how to setup a basic workflow using
``julearn``'s :func:`.run_cross_validation`. There you can find out more about
the required data, building a basic pipeline and how to evaluate your model's
performance.

.. toctree::
   :maxdepth: 2

   cross_validation.rst
   data.rst
   pipeline.rst
   model_evaluation.rst
   model_comparison.rst

If you are just interested in seeing all parameters of
:func:`.run_cross_validation`, click on the function link to have a look at all
its parameters in the :ref:`api`.

If you are already familiar with how to set up a basic workflow using
``julearn`` and want to do more fancy stuff, go to
:ref:`selected_deeper_topics`.
