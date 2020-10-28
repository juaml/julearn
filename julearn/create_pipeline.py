from .components import (
    make_ExtendedDataFrameTranfromer)


def create_pipeline(preprocess_steps_features,
                    preprocess_transformer_target,
                    preprocess_steps_confounds,
                    model_tuple, confounds, categorical_features):

    X_steps = list(preprocess_steps_features) + [model_tuple]
    y_transformer = preprocess_transformer_target
    return make_ExtendedDataFrameTranfromer(X_steps, y_transformer,
                                            preprocess_steps_confounds,
                                            confounds, categorical_features)
