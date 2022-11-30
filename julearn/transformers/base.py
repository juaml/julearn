import pandas as pd
from sklearn.base import TransformerMixin
from .. base import JuBaseEstimator


class JuTransformer(JuBaseEstimator, TransformerMixin):

    def _add_backed_filtered(self, X, X_trans):
        filtered_columns = self._filter(X)
        non_filtered_columns = [
            col
            for col in list(X.columns)
            if col not in filtered_columns

        ]
        return pd.concat(
            (X[:, non_filtered_columns], X_trans),
            axis=1
        )
