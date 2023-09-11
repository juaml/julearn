# Authors: Vera Komeyer <v.komeyer@fz-juelich.de>
#          Fede Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL
"""
Cross-validation consistent Confound Removal
============================================

In many machine learning applications, researchers ultimately want to assess
whether the features are related to the target. However, in most real-world
scenarios the supposed relationship between the features and the target
may be confounded by one or more (un)observed variables. Therefore, the effect
of potential confounding variables is often removed by training a linear
regression to predict each feature given the confounds, and using the residuals
from this confound removal model to predict the target [#1]_, [#2]_. Similarly,
one may instead remove the confounding effect by performing confound regression
on the target. That is, one may predict the target given the confounds, and
then predict the residuals from such a confound removal model using the
features [#3]_. In either case, it is important that such confound regression
models are trained within the cross-validation splits, rather than on the
training and testing data jointly in order to prevent test-to-train data
leakage [#4]_, [#5]_.

Confound Removal in ``julearn``
-------------------------------

``julearn`` implements cross-validation consistent confound regression for both
of the scenarios laid out above (i.e., either confound regression on the
features or on the target) allowing the user to implement complex machine
learning pipelines with relatively little code while avoiding test-to-train
leakage during confound removal.

Let us initially consider removing a confounding variable from the features.

Removing Confounds from the Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first scenario involves confound regression on the features. In order to
do this we can simply configure an instance of a :class:`.PipelineCreator`
by adding the ``"confound_removal"`` step.

We can create some data using ``scikit-learn``'s
:func:`~sklearn.datasets.make_regression`
and then simulate a normally distributed random variable that has a linear
relationship with the target that we can use as a confound.

Let's import some of the functionality we will need:

"""
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator, TargetPipelineCreator
from sklearn.datasets import make_regression

import numpy as np
import pandas as pd

###############################################################################
# First we create the features and the target, and based on this we create
# two artificial confounds that we can use as an example:

# make X and y
X, y = make_regression(n_features=20)

# create two normally distributed random variables with the same mean
# and standard deviation as y
normal_dist_conf_one = np.random.normal(y.mean(), y.std(), y.size)
normal_dist_conf_two = np.random.normal(y.mean(), y.std(), y.size)

# prepare some noise to add to the confounds
noise_conf_one = np.random.rand(len(y))
noise_conf_two = np.random.rand(len(y))

# create the confounds by adding the y and multiplying with a noise factor
confound_one = normal_dist_conf_one + y * noise_conf_one
confound_two = normal_dist_conf_two + y * noise_conf_two

###############################################################################
# Let's organise these data as a :class:`pandas.DataFrame`, which is the
# preferred data format when using ``julearn``:

# put the features into a dataframe
data = pd.DataFrame(X)

# give features and confounds human readable names
features = [f"feature_{x}" for x in data.columns]
confounds = ["confound_1", "confound_2"]

# make sure that feature names and column names are the same
data.columns = features

# add the target to the dataframe
data["my_target"] = y

# add the confounds to the dataframe
data["confound_1"] = confound_one
data["confound_2"] = confound_two

###############################################################################
# In this example, we only distinguish between two types of variables in the
# ``X``. That is, we have 1. our features (or predictors) and 2. our confounds.
# Let's prepare the ``X_types`` dictionary that we hand over to
# :func:`.run_cross_validation` accordingly:

X_types = {"features": features, "confounds": confounds}

###############################################################################
# Now, that we have all the data prepared, and we have defined our ``X_types``,
# we can think about creating the pipeline that we want to run. Now, this is
# the crucial point at which we parametrize the confound removal. We initialize
# the :class:`.PipelineCreator` and add to it as a step using the
# ``"confound_removal"`` transformer (the underlying transformer object is the
# :class:`.ConfoundRemover`).

pipeline_creator = PipelineCreator(
    problem_type="regression", apply_to="features"
)
pipeline_creator.add("confound_removal", confounds="confounds")
pipeline_creator.add("linreg")

print(pipeline_creator)

###############################################################################
# As you can see, we tell the :class:`.PipelineCreator` that we want to work on
# a "regression" problem when we initialize the class. We also tell that by
# default each "step" of the pipeline should be applied to the features whose
# type is ``"features"``. In the first step that we add, we specify we want to
# perform ``"confound_removal"``, and that the features that have the type
# ``"confounds"`` should be used as confounds in the confound regression.
# Note, that because we already specified ``apply_to="features"``
# during the initialization, we do not need to explicitly state this again.
# In short, the ``"confounds"`` will be removed from the ``"features"``.
#
# As a second and last step, we simply add a linear regression (``"linreg"``) to
# fit a predictive model to the de-confounded ``X`` and the ``y``.
#
# Lastly, we only need to apply this pipeline in the :func:`.run_cross_validation`
# function to perform confound removal on the features in a cross-validation
# consistent way:

scores = run_cross_validation(
    data=data,
    X=features + confounds,
    y="my_target",
    X_types=X_types,
    model=pipeline_creator,
    scoring="r2",
)

print(scores)

###############################################################################
# Now, what if we want to remove the confounds from the target rather than the
# features instead?
#
# Removing Confounds from the Target
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If we want to remove the confounds from the target rather than from the
# features, we need to create a slightly different pipeline. ``julearn`` has a
# specific :class:`.TargetPipelineCreator` to perform transformations on the
# target. We first configure this pipeline and add the ``"confound_removal"``
# step.

target_pipeline_creator = TargetPipelineCreator()
target_pipeline_creator.add("confound_removal", confounds="confounds")

print(target_pipeline_creator)

###############################################################################
# Now we insert the target pipeline into the main pipeline that will be used
# to do the prediction. Importantly, we specify that the target pipeline should
# be applied to the ``"target"``.

pipeline_creator = PipelineCreator(
    problem_type="regression", apply_to="features"
)
pipeline_creator.add(target_pipeline_creator, apply_to="target")
pipeline_creator.add("linreg")

print(pipeline_creator)

###############################################################################
# Having configured this pipeline, we can then simply use the same
# :func:`.run_cross_validation` call to obtain our results:

scores = run_cross_validation(
    data=data,
    X=features + confounds,
    y="my_target",
    X_types=X_types,
    model=pipeline_creator,
    scoring="r2",
)

print(scores)

###############################################################################
# As you can see, applying confound regression in your machine learning
# pipeline in a cross-validated fashion is reasonably easy using ``julearn``.
# If you are considering whether or not to use confound regression, however,
# there are further important considerations:
#
# Should I use Confound Regression?
# ---------------------------------
#
# One reason why one might want to perform confound regression in a machine
# learning pipeline is to account for the effects of the confounding variables
# on the target. This can help to mitigate the potential bias introduced by the
# confounding variables and provide more accurate estimates of the true
# relationship between the features and the target.
#
# On the other hand, some argue that confound regression may not always be
# necessary or appropriate, as it can lead to loss of valuable information in
# the data. Additionally, confounding variables may sometimes be difficult to
# identify or measure accurately, which can make confound regression
# challenging or ineffective. In particular, controlling for some variables
# that are not confounds, but in fact colliders, may introduce spurious
# relationships between your features and your targets [#6]_. Lastly, there is
# also some evidence that removing confounds can leak information about the
# target into the features, biasing the resulting predictive models [#7]_.
# Ultimately, the decision to perform confound regression in a machine learning
# pipeline should be based on careful consideration of the specific dataset
# and research question at hand, as well as a thorough understanding of the
# strengths and limitations of this technique.
#
#
# .. topic:: References:
#
#   .. [#1] Rao, Anil, et al., `"Predictive modelling using neuroimaging data \
#      in the presence of confounds" \
#      <https://www.sciencedirect.com/science/article/pii/S1053811917300897>`_,
#      NeuroImage, Volume 150, 15 April 2017, Pages 23-49
#
#   .. [#2] Snoek, Lukas, et al., `"How to control for confounds in decoding \
#      analyses of neuroimaging data" \
#      <https://www.sciencedirect.com/science/article/pii/S1053811918319463?
#      via%3Dihub>`_, NeuroImage, Volume 184, 1 January 2019, Pages 741-760
#
#   .. [#3] He, Tong, et al., `"Deep neural networks and kernel regression \
#      achieve comparable accuracies for functional connectivity prediction \
#      of behavior and demographics" <https://www.sciencedirect.com/science/\
#      article/pii/S1053811919308675>`_, NeuroImage, Volume 206, 1 February
#      2020, 116276
#
#   .. [#4] More, Shammi, et al., `"Confound Removal and Normalization in \
#      Practice: A Neuroimaging Based Sex Prediction Case Study" \
#      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7903939/>`_, Machine
#      Learning and Knowledge Discovery in Databases. Applied Data Science and
#      Demo Track. 2021 Jan 30; 12461: 3–18.
#
#   .. [#5] Chyzhyk, Darya et al., `"How to remove or control confounds in \
#      predictive models, with applications to brain biomarkers" \
#      <https://pubmed.ncbi.nlm.nih.gov/35277962/>`_, Gigascience, 2022 Mar 12.
#
#   .. [#6] Holmberg, Mathias J. `"Collider Bias" \
#      <https://jamanetwork.com/journals/jama/fullarticle/2790247>`_, JAMA.
#      2022; 327 (13):1282-1283. doi:10.1001/jama.2022.1820
#
#   .. [#7] Hamdan, Sami et al., `"Confound-leakage: Confound Removal in \
#      Machine Learning Leads to Leakage" <https://arxiv.org/abs/2210.09232>`_,
#      arXiv:2210.09232, last revised 27 Oct 2022
