from julearn.utils import raise_error


def preprocess(pipeline, X, df, until=None):

    _X = df[X]
    if until is None:
        return pipeline[:-1].transform(df[_X])

    for i, (name, _) in enumerate(pipeline.steps[:-1]):
        if name.replace("wrapped_", "") == until.replace("wrapped_", ""):
            break
    else:
        raise_error(f"No {until} found.")
    return pipeline[:i+1].transform(_X)
