.. include:: links.inc
.. include:: changes/contributors.inc

.. _whats_new:

What's new
==========

.. towncrier release notes start

Julearn 0.3.1 (2024-01-23)
--------------------------

Bugfixes
^^^^^^^^

- Add ``BaseEstimator`` to ``ConfoundRemoval`` for Target by `Sami Hamdan`_
  (:gh:`151`)
- Skip wrapping scorers when target transformers can inverse transform and
  avoid logging filtered warnings by `Fede Raimondo`_ (:gh:`236`)
- Fix `alternative` parameter not being used in the corrected t-test `Fede
  Raimondo`_ (:gh:`245`)
- Fix hyperparameter not being set in `PipelineCreator.add` when an object is
  used as model, by `Fede Raimondo`_ (:gh:`247`)


Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

- Fix cross-referencing links in documentation by `Synchon Mandal`_ (:gh:`168`)
- Standardize ``Sphinx`` section and header ordering format by `Synchon
  Mandal`_ (:gh:`234`)
- Improve documentation language, fix typos and code snippets by `Synchon
  Mandal`_ (:gh:`235`)


Enhancements
^^^^^^^^^^^^

- Make CBPM use sum by default by `Sami Hamdan`_ (:gh:`170`)


Features
^^^^^^^^

- Add support for multiple grids in hyperparameter tuning by `Fede Raimondo`_
  (:gh:`47`)


Misc
^^^^

- Adopt ``ruff`` as the only linter for the codebase by `Synchon Mandal`_
  (:gh:`231`)
- Improve ``codespell`` support and fix typos in documentation by `Synchon
  Mandal`_ (:gh:`232`)
- Adopt ``pre-commit`` for adding and managing git pre-commit hooks by `Synchon
  Mandal`_ (:gh:`233`)

Julearn 0.3.0 (2023-07-19)
--------------------------

The general API and behavior of Julearn has been changed to make it easier to
define and use pipelines. The documentation has been updated to reflect these
changes. Basic functionality from the previous API is still 
present. However, it might require setting a few more parameters in the
:func:`.run_cross_validation` function.

Julearn 0.2.5 (2022-07-21)
--------------------------

API Changes
^^^^^^^^^^^

- Make API surrounding registering consistently use overwrite by `Sami Hamdan`_
- Inner ``cv`` needs to be provided using `search_params`. Deprecating `cv` in
  `model_params` by `Sami Hamdan`_ (:gh:`146`)
- Add ``n_jobs`` and ``verbose`` to ``run_cross_validation`` by `Sami Hamdan`_


Bugfixes
^^^^^^^^

- Fix a hyperparameters setting issue where the parameter had an iterable of
  only one element by `Sami Hamdan`_ (:gh:`96`)
- Fix installations instruction for latest development version (add ``--pre``)
  by `Fede Raimondo`_
- Fix target transformers that only normal transformers are wrapped by
  `Sami Hamdan`_ (:gh:`94`)
- Fix compatibility with new scikit-learn release 0.24 by `Sami Hamdan`_
  (:gh:`108`)
- Fix compatibility with multiprocessing in scikit-learn by `Sami Hamdan`_
- Raise error message when columns in the dataframe are nos strings by
  `Fede Raimondo`_ (:gh:`77`)
- Fix not implemented bug for decision_function in ExtendedDataFramePipeline by
  `Sami Hamdan`_ (:gh:`135`)
- Fix Bug in the transformer wrapper implementation by `Sami Hamdan`_
  (:gh:`122`)
- Fix Bug Target Transformer missing BaseEstimator by `Sami Hamdan`_ (:gh:`151`)
- Fix Bug of showing Warnings when using confound removal by `Sami Hamdan`_
  (:gh:`152`)
- Fix Bug registered scorer not working in dictionary for scoring by `Sami Hamdan`_


Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

- Add *What's new* section in DOC to document changes by `Fede Raimondo`_
- Add information on updating the *What's new* section before releasing by
  `Fede Raimondo`_
- Update docs to make it more uniform by `Kaustubh Patil`_
- Add example for ``tranform_until`` by `Shammi More`_ (:gh:`63`)
- Add documentation/example for parallelization by `Sami Hamdan`_


Enhancements
^^^^^^^^^^^^

- Bump minimum python version to 3.7 by `Fede Raimondo`_
- Refactor scoring to allow for registering and callable scorers by
  `Sami Hamdan`_
- Update :mod:`.model_selection` and add capabilities to register searchers by
  `Sami Hamdan`_
- Update default behavior of setting inner cv according to scikit-learn instead
  of using outer cv as default by `Sami Hamdan`_
- Add tests and more algorithms to ``DynamicSelection`` by `Sami Hamdan`_ and
  `Shammi More`_

Features
^^^^^^^^

- Add user facing ``create_pipeline`` function by `Sami Hamdan`_
- Add CV schemes for stratifying continuous variables, useful for
  regression problems. Check :class:`.ContinuousStratifiedKFold` and
  :class:`.RepeatedContinuousStratifiedKFold` by
  `Fede Raimondo`_ and `Shammi More`_
- Add ``CBPM`` transformer by `Sami Hamdan`_
- Add ``register_model`` by `Sami Hamdan`_ (:gh:`105`)
