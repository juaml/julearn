.. include:: links.inc

Main Components
===============
Julearn aims to provide an user-friendley :doc:`api <api>`. 
In order to use this api you do not need to understand the internals
of julearn.  Still, for some usecases it might be beneficial to use the 
internal classes of Julearn to customize your machine learning pipeline even 
more. This page gives you and overview and introduction to the internal
structure of Julearn and provides a way how to use it as an advanced api for 
machine learning. 

Overview
********
Julearn is based on `scikit-learn`_ , but adds functionality
to deal with confounds and transform the ground truth (the target) inside of 
cross-validaton. To deal with these problems we created the following key 
components and a Column Type System:  


- :class:`ExtendedDataFramePipeline` combines all the other components to
  create a coherent pipeline, which can be used inside of cross-validation.
  You can use :func:`.create_extended_pipeline` as a convinience function to 
  create an :class:`.ExtendedDataFramePipeline`.

  .. note::
    When using customized scorerers together with the 
    :class:`.ExtendedDataFramePipeline` you need to use Julearn's 
    :func:`.get_extended_scorer`.

- DataFramePipeline created using :func:`.create_dataframe_pipeline`. 
  A sklearn pipeline which can be used with `pandas.DataFrame`_ and is able to 
  apply different transformers to different sets of columns
  inside of the DataFrame. The ExtendedDataFramePipeline uses such 
  DataFramePipelines to transform confounds, features and subsets of both 
  seperatley. 

- :class:`DataFrameTransformer` takes in a normal scikit-learn compatible 
  transformer and returns one which can be applied to a subset of features and 
  confounds of one `pandas.DataFrame`_. For the fit and transform methods it 
  always takes in a `pandas.DataFrame`_ and always returns one as well.  
  The DataFramePipeline uses :class:`DataFrameTransformer`\s for each
  transformer inside of it. 

Column Type System
******************

Context
^^^^^^^
To be able to discriminate between different types of variables Julearn
uses a Column Type System. This system currently distinguies between 
continuous variables/features, categorical variables/features and confounds.

.. note::
  On most levels of Julearn this Column Type System is only used internally. 
  Therefore, users do not have to work with it directley.
  For example, by providing the confounds and categorical variables to the
  :class:`.ExtendedDataFramePipeline` it has all the information needed to 
  apply the Column Type Sytem internally without any further input or changes
  to the `pandas.DataFrame`.

How it works
^^^^^^^^^^^^
Every `pandas.DataFrame`_ column has a column name. 
Inside of Julearn we add another string containing the type of the column 
seperated by our delimiter: '__:type:__' to the original column names. 
For example: 

  * We have the original columns :

    - 'Intelligence'
    - 'Age' 
    - 'LikesEvoks'

  * We know:

    - Intelligence is a **continuous** variable 
    - Age is a **confound** 
    - LikesEvoks is a **categorical** variable.
      Either someone likes Evoks or not.
  * Inside of Julearn's Column Type System we can provide this information
    by changing the column names to:
      
      - 'Intelligence\_\_:type:\_\_continuous'
      - 'Age\_\_:type:\_\_confound'
      - 'LikesEvoks\_\_:type:\_\_categorical'

ExtendedDataFramePipeline
*************************
.. autoclass:: julearn.pipeline.ExtendedDataFramePipeline
.. autofunction:: julearn.pipeline.create_extended_pipeline

DataFramePipeline
*****************
.. autofunction:: julearn.pipeline.create_dataframe_pipeline

DataFrameTransformer
********************
.. autoclass:: julearn.transformers.DataFrameTransformer