

Why cross validation?
=====================

One of the 

Ressources to cite:
From https://www.sciencedirect.com/science/article/pii/S105381191630595X:
cross-validation, the standard tool to measure predictive power and tune parameters in decoding.
Read this paper particularly the section
"A primer on cross-validation" to understand ... 
... important concepts in cross validation for decoding from brain images.

Mostly:
https://www.nature.com/articles/s41746-022-00592-y
very summarized why CV is good in section:
"Improper evaluation procedures and leakage"


Maybe:
https://www.sciencedirect.com/science/article/pii/S1053811917305311

scikitlearn:
https://scikit-learn.org/stable/modules/cross_validation.html


The essence of :func:`.run_cross_validation`
--------------------------------------------

:func:`.run_cross_validation` uses your **specified parameters** to train a model 
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
the model(s) of the performed cross validation are 
(depending on your exact specifications).
