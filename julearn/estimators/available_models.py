from sklearn.svm import SVC, SVR
from sklearn.dummy import DummyClassifier, DummyRegressor

available_models = {
    'svm': {
        'regression': SVR(),
        'binary_classification': SVC(),
        'multiclass_classification': SVC()
    },
    'dummy': {
        'regression': DummyRegressor(),
        'binary_classification': DummyClassifier(),
        'multiclass_classification': DummyClassifier(),
    },
}
