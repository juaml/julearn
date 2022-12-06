from julearn.utils import raise_error


def preprocess(pipeline, X, data, until=None, with_column_types=False):

    _X = data[X]
    if until is None:
        return pipeline[:-1].transform(_X)

    for i, (name, _) in enumerate(pipeline.steps[:-1]):
        if name.replace("wrapped_", "") == until.replace("wrapped_", ""):
            break
    else:
        raise_error(f"No {until} found.")
    df_out = pipeline[:i+1].transform(_X)

    if not with_column_types:
        df_out = df_out.rename(columns=lambda col: col.split("__:type:__")[0])
    return df_out
