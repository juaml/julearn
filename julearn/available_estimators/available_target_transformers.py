from ..components.compose_transformers import TargetTransfromerWrapper
from sklearn.preprocessing import StandardScaler
from .custom_transformers import TargetPassThroughTransformer

available_target_transformers = {
    'z_score': TargetTransfromerWrapper(StandardScaler()),
    'passthrough': TargetPassThroughTransformer()

}
