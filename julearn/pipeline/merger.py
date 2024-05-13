"""Module to merge multiple pipelines into a single one."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>prepa
# License: AGPL

from typing import Dict

from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from ..model_selection.available_searchers import (
    get_searcher,
    get_searcher_params_attr,
)
from ..prepare import prepare_search_params
from ..utils.logging import raise_error
from ..utils.typing import EstimatorLike
from .pipeline_creator import _prepare_hyperparameter_tuning


def merge_pipelines(
    *pipelines: EstimatorLike, search_params: Dict
) -> Pipeline:
    """Merge multiple pipelines into a single one.

    Parameters
    ----------
    pipelines : List[EstimatorLike]
        List of estimators that will be merged.
    search_params : Dict
        Dictionary with the search parameters.

    Returns
    -------
    merged : BaseSearchCV
        The merged pipeline as a searcher.

    """

    # Check that we only merge pipelines and searchers. And if there is a
    # searcher, they are all of the same kind and match the search params.

    search_params = prepare_search_params(search_params)
    t_searcher = get_searcher(search_params["kind"])
    for p in pipelines:
        if not isinstance(p, (Pipeline, BaseSearchCV)):
            raise_error(
                "Only pipelines and searchers are supported. "
                f"Found {type(p)} instead."
            )

        if isinstance(p, BaseSearchCV):
            if not isinstance(p, t_searcher):  # type: ignore
                raise_error(
                    "One of the pipelines to merge is a "
                    f"{p.__class__.__name__}, but the search params specify a "
                    f"{search_params['kind']} search. "
                    "These pipelines cannot be merged."
                )
    # Check that all estimators have the same named steps in their pipelines.
    reference_pipeline = pipelines[0]
    if isinstance(reference_pipeline, BaseSearchCV):
        reference_pipeline = reference_pipeline.estimator  # type: ignore

    step_names = reference_pipeline.named_steps.keys()  # type: ignore

    for p in pipelines:
        if isinstance(p, BaseSearchCV):
            p = p.estimator  # type: ignore
            if not isinstance(p, Pipeline):
                raise_error("All searchers must use a pipeline.")
        if step_names != p.named_steps.keys():  # type: ignore
            raise_error("All pipelines must have the same named steps.")

    # The idea behind the merge is to create a list of parameter
    # grids/distributions from a list of pipeline and searchers, to then
    # wrap them into a single searcher. Since all the searchers have the same
    # steps, this is possible. We just need to concatenate the
    # grids/distributions from all searchers. If one of the pipelines is not
    # a searcher, then this means that it has no hyperparameters to tune, but
    # the pipeline is one of the hyperparameter options.

    different_steps = []
    for t_step_name in step_names:
        # Get the transformer/model of the first element
        t = reference_pipeline.named_steps[t_step_name]  # type: ignore

        # Check that all searchers have the same transformer/model.
        # TODO: Fix this comparison, as it always returns False.
        for s in pipelines[1:]:
            if isinstance(s, BaseSearchCV):
                if s.estimator.named_steps[t_step_name] != t:  # type: ignore
                    different_steps.append(t_step_name)
                    break
            else:
                if s.named_steps[t_step_name] != t:  # type: ignore
                    different_steps.append(t_step_name)
                    break

    # Then, we will update the grid of the searchers that have different
    # transformer/model.
    all_grids = []
    for s in pipelines:
        if isinstance(s, BaseSearchCV):
            params_attr = get_searcher_params_attr(s.__class__)
            if params_attr is None:
                raise_error(
                    f"Searcher {s.__class__.__name__} is not registered "
                    "in the searcher registry. Merging of these kinds of "
                    "searchers is not supported. If you register the "
                    "searcher, you can merge it."
                )
            t_grid = getattr(s, params_attr).copy()
        else:
            t_grid = {}
        for t_name in different_steps:
            if isinstance(s, BaseSearchCV):
                t_grid[t_name] = [
                    s.estimator.named_steps[t_name]  # type: ignore
                ]
            else:
                t_grid[t_name] = [s.named_steps[t_name]]  # type: ignore
        all_grids.append(t_grid)

    # Finally, we will concatenate the grids and create a new searcher.
    new_searcher = _prepare_hyperparameter_tuning(
        all_grids, search_params, reference_pipeline  # type: ignore
    )
    return new_searcher
