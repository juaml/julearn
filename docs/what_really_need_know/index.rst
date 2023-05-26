.. include:: ../links.inc

.. _need_to_know:

############################
What you really need to know
############################


The backbone of Julearn is the function :func:`.run_cross_validation`, which
allows you to do all the *magic*. All important information needed for your 
machine learning workflow goes into this function, specified via its parameters.

But why is basically everything based on one `cross_validation` function? Well,
because doing proper cross-validation is the backbone of machine learning. To 
understand why, reading the sub-chapter :ref:`cross_validation` ist a good starting
point.

Once you are familiar with the importance of cross-validation, to see what is 
needed to setup a basic workflow using Julearn's :func:`.run_cross_validation`, 
function, please follow along the other sub-chapters of this chapter.
There you can find out more about the data input, a basic pipeline and
basic model evaluation steps:

.. toctree::
   :maxdepth: 2

   cross_validation.rst
   data.rst
   pipeline.rst
   model_evaluation.rst

If you are just interested in seeing all parameters of :func:`.run_cross_validation`, 
click on the function link to have a look at all its parameters in the `api`.

If you are already familiar with how to set up a basic workflow using Julearn 
and want to do more fancy stuff, go to :ref:`selected_deeper_topics` to pick 
and select specific topics.