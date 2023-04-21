# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource, FactorRange, Whisker
from bokeh.palettes import Colorblind
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, jitter


SCORE_PLOT_TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"


class _JulearnScoresViewer(param.Parameterized):
    """
    A class to visualize the scores for model comparison.

    Parameters
    ----------
    long_df : pd.DataFrame
        A long dataframe with the scores.
    width : int
        The width of the plot.
    height : int
        The height of the plot.
    ci : int
        The confidence interval to use for the error bars.
    """

    metric = param.Selector(None, default=None)
    models = param.ListSelector(default=None, objects=None)
    show_train = param.Boolean(False)
    group_repeats = param.Selector(
        objects=["mean", "median", "no"], default="no"
    )
    _long_df = None
    _width = 800
    _height = 400
    _ci = 95

    def set_data(self, long_df):
        """Set the data to use for the plot.

        Parameters
        ----------
        long_df : pd.DataFrame
            A long dataframe with the scores. Must have the columns
            "model", "metric", "score", "fold", "repeat", "set".w

        Returns
        -------
        self : _JulearnScoresViewer
        """
        self._long_df = long_df
        self.param.metric.objects = long_df["metric"].unique().tolist()
        self.param.models.objects = long_df["model"].unique().tolist()
        self.param.metric.default = self.param.metric.objects[0]
        self.param.models.default = self.param.models.objects
        self.metric = self.param.metric.default
        self.models = self.param.models.default
        return self

    def set_width(self, width):
        """Set the width of the plot.

        Parameters
        ----------
        width : int
            The width of the plot.

        Returns
        -------
        self : _JulearnScoresViewer
        """
        self._width = width
        return self

    def set_height(self, height):
        """Set the height of the plot.

        Parameters
        ----------
        height : int
            The height of the plot.

        Returns
        -------
        self : _JulearnScoresViewer
        """
        self._height = height
        return self

    def set_ci(self, ci):
        """Set the CI to use for the error bars.

        Parameters
        ----------
        ci : float
            The CI to plot.

        Returns
        -------
        self : _JulearnScoresViewer
        """
        self._ci = ci
        return self

    @param.depends("metric", "show_train", "group_repeats", "models")
    def plot_scores(self):
        t_metric = self.metric
        show_train = self.show_train
        group_repeats = self.group_repeats
        t_df = self._long_df[self._long_df["metric"] == t_metric]
        if group_repeats != "no":
            t_group = t_df.groupby(["model", "set", "metric", "repeat"])
            if group_repeats == "mean":
                t_df = t_group.mean()["score"].reset_index()
            elif group_repeats == "median":
                t_df = t_group.median()["score"].reset_index()
            t_df["fold"] = "all"
        _models = self.models
        if "train" in self._long_df["set"].unique() and show_train:
            sets = ["train", "test"]
            x = [(m, s) for m in _models for s in sets]
            x_values = list(map(tuple, t_df[["model", "set"]].values))
            color = factor_cmap(
                "x", palette=Colorblind[3], factors=sets, start=1, end=3
            )
        else:
            sets = ["test"]
            x = [m for m in _models]
            t_df = t_df[t_df["set"] == "test"]
            x_values = list(t_df["model"].values)
            color = "grey"

        x_range = FactorRange(factors=x)
        scores = t_df["score"].values
        folds = t_df["fold"].values
        repeats = t_df["repeat"].values
        set_values = t_df["set"].values
        data_dict = {
            "x": x_values,
            "score": scores,
            "fold": folds,
            "repeat": repeats,
            "set": set_values,
        }

        source = ColumnDataSource(data=data_dict)
        # x = [tuple(x) for x in df_results[["file_kind", "storage_kind"]].values]
        p = figure(
            width=self._width,
            height=self._height,
            x_range=x_range,
            title=t_metric,
            tools=SCORE_PLOT_TOOLS,
            tooltips=[
                ("Fold", "@fold"),
                ("Repeat", "@repeat"),
                ("score", "@score"),
            ],
        )
        p.circle(
            x=jitter("x", width=0.7, range=p.x_range),
            y="score",
            alpha=0.5,
            source=source,
            size=10,
            line_color="white",
            color=color,
            legend_group="set",
        )
        if len(sets) > 1:
            p.add_layout(p.legend[0], "right")
        else:
            p.legend.visible = False
        # Add whiskers for CI
        if len(sets) > 1:
            g = t_df.groupby(["model", "set"])
        else:
            g = t_df.groupby("model")
        ci_upper = self._ci + (1 - self._ci) / 2
        ci_lower = (1 - self._ci) / 2
        upper = g.score.quantile(ci_upper)
        lower = g.score.quantile(ci_lower)
        source = ColumnDataSource(
            data=dict(base=upper.index.values, upper=upper, lower=lower)
        )
        error = Whisker(
            base="base",
            upper="upper",
            lower="lower",
            source=source,
            level="annotation",
            line_width=2,
        )
        error.upper_head.size = 20
        error.lower_head.size = 20
        p.add_layout(error)

        # Add whiskers for mean
        mean_score = g.score.mean()
        source = ColumnDataSource(
            data=dict(
                base=mean_score.index.values,
                upper=mean_score,
                lower=mean_score,
            )
        )
        mean_bar = Whisker(
            base="base",
            upper="upper",
            lower="lower",
            source=source,
            level="annotation",
            line_width=2,
        )
        mean_bar.upper_head.size = 10
        mean_bar.lower_head.size = 10
        p.add_layout(mean_bar)

        # Add horizontal lines to separate the models

        if len(sets) > 1:
            grp_pad = p.x_range.group_padding
            span_x = [
                tuple(list(t_x) + [1 + (grp_pad - 1.0) / 2.0])
                for t_x in x[1 : -1 : len(sets)]
            ]
        else:
            grp_pad = p.x_range.factor_padding
            span_x = [(t_x, 1 + (grp_pad - 1.0) / 2.0) for t_x in x[:-1]]

        src_sep = ColumnDataSource(
            data={
                "base": span_x,
                "lower": [0] * len(span_x),
                "upper": [max(scores) * 1.05] * len(span_x),
            }
        )
        sep = Whisker(
            base="base",
            upper="upper",
            lower="lower",
            source=src_sep,
            level="annotation",
            line_width=2,
            line_color="lightgrey",
            dimension="height",
            line_alpha=1,
            upper_head=None,
            lower_head=None,
        )
        p.add_layout(sep)

        # show(p)
        # p.y_range.start = 0
        if len(sets) > 1:
            p.xaxis.major_tick_line_color = None
            p.xaxis.major_label_text_font_size = "0pt"
            p.xaxis.group_label_orientation = "vertical"
        else:
            p.xaxis.major_label_orientation = "vertical"
            p.xaxis.major_label_text_color = "grey"
        p.xgrid.grid_line_color = None

        return p


def plot_scores(
    *scores: pd.DataFrame,
    width: int = 800,
    height: int = 600,
    ci: float = 0.95,
) -> pn.layout.Panel:
    """Plot the scores of the models on a panel dashboard.

    Parameters
    ----------
    *scores : pd.DataFrame
        DataFrames containing the scores of the models. The DataFrames must
        be the output of `run_cross_validation`
    width : int, optional
        Width of the plot (default is 800)
    height : int, optional
        Height of the plot (default is 600)
    ci : float, optional
        Confidence interval to use for the plots (default is 0.95)
    """
    # Transform the data for plotting
    results_df = pd.concat([*scores])
    all_metrics = [
        x
        for x in results_df.columns
        if x.startswith("test_") or x.startswith("train_")
    ]

    long_df = results_df.set_index(["model", "fold", "repeat"])[
        all_metrics
    ].stack()
    long_df.name = "score"
    long_df.index.names = ["model", "fold", "repeat", "metric"]
    long_df = long_df.reset_index()

    long_df["set"] = long_df["metric"].str.split("_").str[0]
    long_df["metric"] = (
        long_df["metric"].str.replace("train_", "").str.replace("test_", "")
    )
    all_sets = long_df["set"].unique().tolist()

    viewer = _JulearnScoresViewer().set_data(long_df).set_width(width)
    pn.extension(template="fast", theme="dark")
    dashboard_title = pn.panel("## Scores Viewer")
    png = pn.panel("./julearn_logo_it.png", width=200)
    header = pn.Row(png, pn.Spacer(width=50), dashboard_title)
    widget_row = pn.Row(
        pn.Param(
            viewer.param.metric,
            name="Metric",
            show_name=True,
            widgets={
                "metric": {
                    "type": pn.widgets.Select,
                    "button_type": "primary",
                    "name": "",
                }
            },
        )
    )
    if len(all_sets) > 1:
        widget_row.append(
            pn.Param(
                viewer.param.show_train,
                name="Train Scores",
                show_name=True,
                widgets={
                    "show_train": {
                        "type": pn.widgets.Toggle,
                        "button_type": "primary",
                        "name": "Show",
                    }
                },
            )
        )

    widget_row.append(
        pn.Param(
            viewer.param.group_repeats,
            name="Aggregate Repeats",
            show_name=True,
            widgets={
                "group_repeats": {
                    "type": pn.widgets.RadioButtonGroup,
                    "button_type": "primary",
                    "options": ["no", "median", "mean"],
                }
            },
        )
    )

    models_widget = pn.Param(
        viewer.param.models,
        name="Models",
        show_name=True,
        widgets={
            "models": {
                "type": pn.widgets.CheckButtonGroup,
                "button_type": "primary",
                "orientation": "vertical",
            }
        },
    )

    column = pn.Column(
        header,
        widget_row,
        pn.Row(
            viewer.plot_scores,
            models_widget,
        ),
    )
    return column
