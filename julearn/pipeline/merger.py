"""Module to merge multiple pipelines into a single one."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>prepa
# License: AGPL

from typing import Dict

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..utils.logging import raise_error
from ..utils.typing import EstimatorLike
from .pipeline_creator import _prepare_hyperparameter_tuning
from ..prepare import prepare_search_params


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

    for p in pipelines:
        if not isinstance(p, (Pipeline, GridSearchCV, RandomizedSearchCV)):
            raise_error(
                "Only pipelines and searchers are supported. "
                f"Found {type(p)} instead.")
        if isinstance(p, GridSearchCV):
            if search_params["kind"] != "grid":
                raise_error(
                    "At least one of the pipelines to merge is a "
                    "GridSearchCV, but the search params do not specify a "
                    "grid search. These pipelines cannot be merged."
                )
        elif isinstance(p, RandomizedSearchCV):
            if search_params["kind"] != "random":
                raise_error(
                    "At least one of the pipelines to merge is a "
                    "RandomizedSearchCV, but the search params do not specify "
                    "a random search. These pipelines cannot be merged."
                )

    # Check that all estimators have the same named steps in their pipelines.
    reference_pipeline = pipelines[0]
    if isinstance(reference_pipeline, (GridSearchCV, RandomizedSearchCV)):
        reference_pipeline = reference_pipeline.estimator

    step_names = reference_pipeline.named_steps.keys()

    for p in pipelines:
        if isinstance(p, (GridSearchCV, RandomizedSearchCV)):
            p = p.estimator
            if not isinstance(p, Pipeline):
                raise_error("All searchers must use a pipeline.")
        if step_names != p.named_steps.keys():
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
        t = reference_pipeline.named_steps[t_step_name]

        # Check that all searchers have the same transformer/model.
        # TODO: Fix this comparison, as it always returns False.
        for s in pipelines[1:]:
            if isinstance(s, (GridSearchCV, RandomizedSearchCV)):
                if s.estimator.named_steps[t_step_name] != t:
                    different_steps.append(t_step_name)
                    break
            else:
                if s.named_steps[t_step_name] != t:
                    different_steps.append(t_step_name)
                    break

    # Then, we will update the grid of the searchers that have different
    # transformer/model.
    all_grids = []
    for s in pipelines:
        if isinstance(s, GridSearchCV):
            t_grid = s.param_grid.copy()
        elif isinstance(s, RandomizedSearchCV):
            t_grid = s.param_distributions.copy()
        else:
            t_grid = {}
        for t_name in different_steps:
            if isinstance(s, (GridSearchCV, RandomizedSearchCV)):
                t_grid[t_name] = [s.estimator.named_steps[t_name]]
            else:
                t_grid[t_name] = [s.named_steps[t_name]]
        all_grids.append(t_grid)

    # Finally, we will concatenate the grids and create a new searcher.
    new_searcher = _prepare_hyperparameter_tuning(
        all_grids, search_params, reference_pipeline
    )
    return new_searcher
