.. include:: ../links.inc

Cross-validation consistent confound removal
============================================

In many machine learning applications, researchers ultimately want to assess
whether the features are related to the target. However, in most real-world
scenarios the supposed relationship between the features and the target
may be confounded by one or more (un)observed variables. Therefore, the effect
of potential confounding variables is often removed by training a linear regression
to predict each feature given the confounds, and using the residuals from this 
confound removal model to predict the target [#1]_, [#2]_. Similarly, one may 
instead remove the confounding effect by performing confound regression on the target.
That is, one may predict the target given the confounds, and then predict the residuals
from such a confound removal model using the features [#3]_. In either case, it
is very important that such confound regression models are only trained on the
training data, rather than on the training and testing data jointly in order to
prevent test-to-train data leakage [#4]_, [#5]_.



.. topic:: References:

	.. [#1] Rao, Anil, et al., `"Predictive modelling using neuroimaging data
	in the presence of confounds" <https://www.sciencedirect.com/science/article/pii/S1053811917300897>`_,
	NeuroImage, Volume 150, 15 April 2017, Pages 23-49

	.. [#2] Snoek, Lukas, et al., `"How to control for confounds in decoding analyses of neuroimaging data"
	<https://www.sciencedirect.com/science/article/pii/S1053811918319463?via%3Dihub>`_,
	NeuroImage, Volume 184, 1 January 2019, Pages 741-760

	.. [#3] He, Tong, et al., `"Deep neural networks and kernel regression achieve
	comparable accuracies for functional connectivity prediction of behavior and demographics" 
	<https://www.sciencedirect.com/science/article/pii/S1053811919308675>`_,
	NeuroImage, Volume 206, 1 February 2020, 116276

	.. [#4] More, Shammi, et al., `"Confound Removal and Normalization in 
	Practice: A Neuroimaging Based Sex Prediction Case Study"
	<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7903939/>`_, 
	Machine Learning and Knowledge Discovery in Databases. Applied Data Science and Demo Track. 2021 Jan 30; 12461: 3–18. 

	.. [#5] Chyzhyk, Darya et al., `"How to remove or control confounds in
	predictive models, with applications to brain biomarkers"
	<https://pubmed.ncbi.nlm.nih.gov/35277962/>`_,
	 Gigascience, 2022 Mar 12.