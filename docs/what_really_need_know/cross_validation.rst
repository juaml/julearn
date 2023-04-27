.. _why_cv:

Why cross validation?
=====================

In Machine Learning, usually one wants to test how well a model _predicts_ 
new data. Therefore, we cannot do predictions using the same data we used 
for _fitting_, as this is the data the model learned from and therefore will 
perform very well, sometimes even too well (overfitting). Thus, we need to have
separate data for fitting and predicting. At the same time, data is a valuable 
ressource in machine learning and one wants to use it as efficeint as possible.
To solve this, we use cross validation - one of the core concepts in machine
learning - to split the data multiple times (so called folds) into a training 
and validation dataset and thereby make most efficient use of the available data.
The core idea is that we want to train (fit) a model on a subset of our data and 
evaluate it on a different subset of our data to see how well the trained 
model generalizes to unseen data. The training dataset is used to 
fit a pipeline (see :ref:`pipeline_usage`), while the validation dataset is 
used to predict the data. The predictions are then compared to the true 
values of the validation dataset, obtaining an estimation of the prediction 
performance of the model. This is done in a repeated manner for all folds 
(splits) and the overview of the scores from all folds give a good estimation 
of the model's generalization performance. To read more about cross validation, 
its functionality and usage and why it is such an
important concept in machine learning, you can have a look at these 
[#1]_ [#2]_ [#3]_ [#4]_ resources.


The essence of :func:`.run_cross_validation`
--------------------------------------------

Building pipelines (see :ref:`pipeline_usage`) within a (nested)
cross-validation scheme, without
accidentally leaking some information between steps can quickly become
complicated and erros are often not-obvious to detect. To make make 
cross-validation less prone for such accidental mistakes and more transparent
for debugging, :func:`.run_cross_validation` is the core of Julearn.
It uses your **specified parameters** to train a model 
accordingly and most importantly does all specified steps in a cross-validation 
consistent manner. This helps to avoid data leakage.

Among others, the parameters passed to :func:`.run_cross_validation` include 
the specification of ...

1. ... your data and feature types (see :ref:`data_usage`)
2. ... your pipeline steps, like the learning algorithm or preprocessing steps 
   to use (see :ref:`pipeline_usage`)
3. ... how the model should be evaluated, like cross validation scheme or the 
   scoring to be used for evaluation (see :ref:`model_evaluation_usage`) 

After training the model, :func:`.run_cross_validation` either **returns** 
only the model's scores from the cross validation or both, the scores and 
the model(s) of the performed cross validation 
(depending on your exact specifications) (see :ref:`model_evaluation_usage`).


.. topic:: References:

      .. [#1] https://www.sciencedirect.com/science/article/pii/S105381191630595X

      .. [#2] https://www.nature.com/articles/s41746-022-00592-y

      .. [#3] https://www.sciencedirect.com/science/article/pii/S1053811917305311

      .. [#4] https://scikit-learn.org/stable/modules/cross_validation.html