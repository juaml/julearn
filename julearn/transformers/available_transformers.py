from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . confounds import DataFrameConfoundRemover
from . basic import PassThroughTransformer
from . target import TargetTransfromerWrapper, TargetPassThroughTransformer

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


available_target_transformers = {
    'z_score': TargetTransfromerWrapper(StandardScaler()),
    'passthrough': TargetPassThroughTransformer()

}
