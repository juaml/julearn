.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like 
   :func:`julearn.api.run_cross_validation`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. _current:

Current (0.2.5.dev)
-------------------

Enhancements
~~~~~~~~~~~~
- Add *What's new* section in DOC to document changes (by `Fede Raimondo`_).

- Add information on updating the *What's new* section before releasing (by `Fede Raimondo`_).

- Update docs to make it more uniform (by `Kaustubh Patil`_). 

- Refactor scoring to allow for registering and callable scorers (by `Sami Hamdan`_).

- Update :mod:`.model_selection` and add capabilities to register searchers (by `Sami Hamdan`_).

- Add user facing `create_pipeline` function (by `Sami Hamdan`_)

- Update default behavior of setting inner cv according to scikit-learn instead of using outer cv as default (by `Sami Hamdan`_).
             
- Add tests and more algorithms to `DynamicSelection` (by `Sami Hamdan`_ and `Shammi More`_).

- Add CV schemes for stratifying based on the grouping variables, useful for regression problems. Check :class:`.StratifiedGroupsKFold` and :class:`.RepeatedStratifiedGroupsKFold` (by `Fede Raimondo`_ and `Shammi More`_).


Bugs
~~~~

- Fix a hyperparameters setting issue where the parameter had an iterable of only one element (:gh:`96` by `Sami Hamdan`_).

- Fix installations instruction for latest development version (add ``--pre`` by `Fede Raimondo`_).

- Fix target transformers that only normal transformers are wrapped (:gh:`94` by `Sami Hamdan`_).

- Fix compatibility with new scikit-learn release 0.24 (:gh:`#108` by `Sami Hamdan`_).

- Fix compatibility with multiprocessing in scikit-learn (by `Sami Hamdan`_).

- Raise error message when columns in the dataframe are nos strings (:gh:`77` by `Fede Raimondo`_).

API changes
~~~~~~~~~~~
