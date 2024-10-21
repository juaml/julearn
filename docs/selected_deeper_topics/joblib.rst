.. include:: ../links.inc


.. _joblib_parallel:

Parallelizing julearn with Joblib
=================================

As with `scikit-learn`_, ``julearn`` allows you to parallelize your code using
`Joblib`_. This can be particularly useful when you have a large dataset or
when you are running a computationally expensive operation that can be easily
computed in parallel.

Without going into details about parallel and distributed computing, the idea
is to split the computation into smaller tasks that can be executed
independently from each other. This way, you can take advantage of multiple
*processors* to do them in parallel.