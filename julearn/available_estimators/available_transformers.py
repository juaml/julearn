from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .custom_transformers import (DataFrameConfoundRemover,
                                  PassThroughTransformer)

"""
a dictionary containing all supported transformers
name : [sklearn transformer,
        str:returns_same_features]
"""

available_transformers = {
    "z_score": [StandardScaler(), 'same'],
    "pca": [PCA(), 'unknown'],
    "remove_confound": [
        DataFrameConfoundRemover(),
        'subset',
    ],
    'passthrough': [PassThroughTransformer(), 'same']
}
