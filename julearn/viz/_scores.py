# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from pathlib import Path

import pandas as pd
import panel as pn
import param
from bokeh.models import (
    ColumnDataSource,
    FactorRange,
    Whisker,
    DataTable,
    ScientificFormatter,
    TableColumn,
    Label,
)
from bokeh.palettes import Colorblind
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, jitter

from ..utils.checks import check_scores_df
from ..stats import corrected_ttest


SCORE_PLOT_TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"


class _JulearnScoresViewer(param.Parameterized):
    """
    A class to visualize the scores for model comparison.

    Parameters
    ----------
    *scores : pd.DataFrame
        DataFrames containing the scores of the models. The DataFrames must
        be the output of `run_cross_validation`
    width : int
        The width of the plot (default is 800).
    height : int
        The height of the plot (default is 600).
    ci : int
        The confidence interval to use for the error bars (default is 0.95).
    """

    metric = param.Selector([], default=None)
    models = param.ListSelector(default=None, objects=[])
    sets = param.ListSelector(default=None, objects=[])
    show_stats = param.Boolean(False)
    group_repeats = param.Selector(
        objects=["mean", "median", "no"], default="no"
    )

    def __init__(self, *params, **kwargs):
        scores = kwargs.pop("scores", None)
        if scores is not None:
            self.set_data(scores)

        self.width = kwargs.pop("width", 800)
        self.height = kwargs.pop("height", 600)
        self.ci = kwargs.pop("ci", 0.95)
        super().__init__(*params, **kwargs)

    def set_data(self, scores):
        """Set the data to use for the plot.

        Parameters
        ----------
        scores : list of pd.DataFrame
            DataFrames containing the scores of the models. The DataFrames must
            be the output of `run_cross_validation`
        Returns
        -------
        self : _JulearnScoresViewer
        """

        # Transform the data for plotting
        self.scores = check_scores_df(*scores, same_cv=False)
        results_df = pd.concat(self.scores)
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
            long_df["metric"]
            .str.replace("train_", "")
            .str.replace("test_", "")
        )
        self.long_df = long_df
        self.param.metric.objects = long_df["metric"].unique().tolist()
        self.param.models.objects = long_df["model"].unique().tolist()
        self.param.sets.objects = long_df["set"].unique().tolist()
        self.param.metric.default = self.param.metric.objects[0]
        self.param.models.default = self.param.models.objects
        self.param.sets.default = self.param.sets.objects
        self.metric = self.param.metric.default
        self.models = self.param.models.default
        self.sets = self.param.sets.default
        return self

    @param.depends("metric", "sets", "group_repeats", "models")
    def plot_scores(self):
        if len(self.sets) == 0:
            p = figure(
                width=self.width,
                height=self.height,
                title=self.metric,
            )
            labels = Label(
                x=self.width / 2,
                y=self.height / 2,
                x_units="screen",
                y_units="screen",
                text_align="center",
                text="Please select a set to display",
                text_font_size="14pt",
            )
            p.add_layout(labels)
            return p
        t_metric = self.metric
        group_repeats = self.group_repeats
        t_df = self.long_df[self.long_df["metric"] == t_metric]
        if group_repeats != "no":
            t_group = t_df.groupby(["model", "set", "metric", "repeat"])
            if group_repeats == "mean":
                t_df = t_group.mean()["score"].reset_index()
            elif group_repeats == "median":
                t_df = t_group.median()["score"].reset_index()
            t_df["fold"] = "all"
        self.models = self.models
        if len(self.sets) > 1:
            x = [(m, s) for m in self.models for s in self.sets]
            x_values = list(map(tuple, t_df[["model", "set"]].values))
            color = factor_cmap(
                "x", palette=Colorblind[3], factors=self.sets, start=1, end=3
            )
        else:
            x = [m for m in self.models]
            t_df = t_df[t_df["set"] == self.sets[0]]
            x_values = list(t_df["model"].values)
            if self.sets[0] == "test":
                color = Colorblind[3][0]
            else:
                color = Colorblind[3][1]

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
        p = figure(
            width=self.width,
            height=self.height,
            x_range=x_range,
            title=t_metric,
            tools=SCORE_PLOT_TOOLS,
            tooltips=[
                ("Fold", "@fold"),
                ("Repeat", "@repeat"),
                ("score", "@score"),
            ],
        )
        if len(self.models) == 0:
            labels = Label(
                x=self.width / 2,
                y=self.height / 2,
                x_units="screen",
                y_units="screen",
                text_align="center",
                text="Please select a at least one model to display",
                text_font_size="14pt",
            )
            p.add_layout(labels)
            return p
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
        if len(self.sets) > 1:
            p.add_layout(p.legend[0], "right")
        else:
            p.legend.visible = False
        # Add whiskers for CI
        if len(self.sets) > 1:
            g = t_df.groupby(["model", "set"])
        else:
            g = t_df.groupby("model")
        ci_upper = self.ci + (1 - self.ci) / 2
        ci_lower = (1 - self.ci) / 2
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

        if len(self.sets) > 1:
            grp_pad = p.x_range.group_padding
            span_x = [
                tuple(list(t_x) + [1 + (grp_pad - 1.0) / 2.0])
                for t_x in x[1 : -1 : len(self.sets)]
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
        if len(self.sets) > 1:
            p.xaxis.major_tick_line_color = None
            p.xaxis.major_label_text_font_size = "0pt"
            p.xaxis.group_label_orientation = "vertical"
        else:
            p.xaxis.major_label_orientation = "vertical"
            p.xaxis.major_label_text_color = "grey"
        p.xgrid.grid_line_color = None

        return p

    @param.depends("metric", "sets", "show_stats")
    def plot_stats(self):
        if self.show_stats and len(self.sets) > 0:
            stats_df = corrected_ttest(*self.scores)
            stats_df["set"] = stats_df["metric"].str.split("_").str[0]
            stats_df["metric"] = (
                stats_df["metric"]
                .str.replace("train_", "")
                .str.replace("test_", "")
            )
            stats_df = stats_df[stats_df["metric"] == self.metric]
            stats_df = stats_df[stats_df["set"].isin(self.sets)]
            stats_df.sort_values(
                by="p-val-corrected", inplace=True, ascending=True
            )
            source = ColumnDataSource(stats_df)
            columns = [
                TableColumn(field="model_1", title="Model 1"),
                TableColumn(field="model_2", title="Model 2"),
                # TableColumn(field="metric", title="Metric"),
                TableColumn(
                    field="t-stat",
                    title="t-stat",
                    formatter=ScientificFormatter(precision=3),
                ),
                TableColumn(
                    field="p-val",
                    title="p-value",
                    formatter=ScientificFormatter(precision=3),
                ),
                TableColumn(
                    field="p-val-corrected",
                    title="corrected_p-value",
                    formatter=ScientificFormatter(precision=3),
                ),
            ]
            if len(self.sets) > 1:
                columns.append(TableColumn(field="set", title="Set"))
            data_table = DataTable(
                source=source,
                columns=columns,
                width=self.width,
                index_position=None,
            )
            return data_table
        else:
            return pn.pane.Markdown("")


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
    viewer = _JulearnScoresViewer(
        scores=[*scores], width=width, height=height, ci=ci
    )
    pn.extension(template="fast")
    dashboard_title = pn.panel("## Scores Viewer")
    logo = Path(__file__).parent / "res" / "julearn_logo_generalization.png"
    png = pn.panel(logo, width=200)
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

    widget_row.append(
        pn.Param(
            viewer.param.show_stats,
            name="Statistics",
            show_name=True,
            widgets={
                "show_stats": {
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

    filter_widgets = pn.Column()
    if len(viewer.sets) > 1:
        filter_widgets.append(
            pn.Param(
                viewer.param.sets,
                name="Sets",
                show_name=True,
                widgets={
                    "sets": {
                        "type": pn.widgets.CheckButtonGroup,
                        "button_type": "primary",
                        "orientation": "vertical",
                    }
                },
            )
        )
    filter_widgets.append(
        pn.Param(
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
    )
    column = pn.Column(
        header,
        widget_row,
        pn.Row(
            viewer.plot_scores,
            filter_widgets,
        ),
        viewer.plot_stats,
    )
    return column
