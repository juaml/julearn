from julearn.utils import raise_error


def preprocess(pipeline, X,  until=None):

    if until is None:
        return pipeline[:-1].transform(X)

    for i, (name, _) in enumerate(pipeline.steps[:-1]):
        if name.replace("wrapped_", "") == until.replace("wrapped_", ""):
            break
    else:
        raise_error(f"No {until} found.")
    return pipeline[:i+1].transform(X)
