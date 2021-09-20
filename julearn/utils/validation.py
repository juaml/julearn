from . logging import raise_error


def is_transformable(t):
    can_transform = hasattr(t, "fit_transform") or hasattr(t, "transform")
    return can_transform


def check_n_confounds(n_confounds):
    if type(n_confounds) != int:
        raise_error(
            f'n_confounds has to be an int, but was {n_confounds}'
        )
    elif n_confounds < 0:
        raise_error(
            f'n_confounds needs to be >=0, but was {n_confounds}.'
        )
