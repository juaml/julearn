import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from julearn.base import WrapModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


@pytest.fixture(
    params=[LinearRegression, LogisticRegression,
            SVR, SVC,
            DecisionTreeRegressor, DecisionTreeClassifier,
            ])
def model(request):
    return request.param


@pytest.mark.parametrize(
    "apply_to,column_types,selection",
    [
        (None, ["continuous"]*4, slice(0, 4),),
        (None, ["continuous"]*3 + ["cat"], slice(0, 3),),
        (["continuous"], ["continuous"]*3 + ["cat"], slice(0, 3),),
        (["cont", "cat"], ["cont"]*3 + ["cat"], slice(0, 4),),
        (None, [""]*4, slice(0, 4),),
        (".*", ["continuous", "duck", "quak", "B"], slice(0, 4),),
        ("*", ["continuous", "duck", "quak", "B"], slice(0, 4),),
    ]
)
def test_WrapModel(X_iris, y_iris, model, apply_to, column_types, selection):

    column_types = [col or "continuous" for col in column_types]
    X_iris.columns = [f"{col.split('__:type:__')[0]}__:type:__{ctype}"
                      for col, ctype in zip(X_iris.columns, column_types)
                      ]
    X_iris_selected = X_iris.iloc[:, selection]

    np.random.seed(42)
    lr = model().fit(X_iris_selected, y_iris)
    pred_sk = lr.predict(X_iris_selected)

    np.random.seed(42)
    wlr = WrapModel(model(), apply_to=apply_to)
    wlr.fit(X_iris, y_iris)
    pred_ju = wlr.predict(X_iris)

    assert_almost_equal(pred_ju, pred_sk)
