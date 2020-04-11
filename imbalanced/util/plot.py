# flake8: noqa

from typing import Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA


class Plot:

    def __init__(
        self,
        image_width: int = 1200,
        image_height: int = 900,
    ) -> None:

        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height

        pio.renderers.default = 'notebook'


    def class_balance(self, class_balance: Dict[str, int]):
        """
        Plots class balance.
        
        Parameters
        ----------
        class_balance : Dict[str, int]
            Record count for each label by name.
        """

        class_balance = dict(sorted(class_balance.items()))

        labels = list(class_balance.keys())
        counts = list(class_balance.values())

        fig = go.Figure()

        bars = go.Bar(
            x=labels,
            y=counts,
            text=counts,
            textposition='auto'
        )

        fig.add_trace(bars)

        fig.update_layout(title='Class balance')
        fig.show()

    # TODO 3d scatter plot
    def features_distribution(self, x: np.ndarray, y: np.ndarray):
        """
        Plots features distribution.
        This method uses PCA algorithm to decompose features space.

        Parameters
        ----------
        x : nd.array
            Records from your data set.
        y : nd.array
            Respective labels for each of the records.
        """

        if x.shape[1] > 2:
            decomposer = PCA(n_components=2)
            records = decomposer.fit(x).transform(x)

        x0 = x[:, 0]
        x1 = x[:, 1]

        y = [str(label) for label in y]

        fig = px.scatter(x=x0, y=x1, color=y)

        fig.update_layout(
            title='Features distribution',
            legend_title='Label'
        )

        fig.show()

    def heatmap(
        self,
        data: np.ndarray,
        plot_name: str,
        ylabel='',
        xlabel=''
    ):
        fig = go.Figure()

        heatmap = go.Heatmap(
            z=data,
            y=ylabel,
            x=xlabel,
            colorscale='Viridis'
        )

        fig.add_trace(heatmap)

        fig.update_layout(
            title=plot_name
        )

        fig.show()

    def scores(self, scores: Dict[str, List[float]]):
        """
        Plots and dumps scores.

        Parameters
        ----------
        metrics : Dict[str, List[float]]
            Grouped scores dictionary, by metric name.
        """

        normalized = {}
        other = {}

        for metric, values in scores.items():
            if np.max(values) <= 1.0:
                normalized[metric] = values
            else:
                other[metric] = values

        if normalized:
            if len(normalized) == 1:
                # TODO: looks ugly, but it's time to sleep
                for metric, values in normalized.items():
                    self.distribution(values, name=metric)

            else:
                self.boxplot(normalized)

        if other:
            for metric, values in other.items():
                self.distribution(values, name=metric)

    def boxplot(self, data: Dict[str, List[float]]):

        fig = go.Figure()

        for metric, values in data.items():

            fig.add_trace(
                go.Box(
                    y=values,
                    name=metric
                )
            )

        fig.show()

    def distribution(self, data: List[float], name: str=''):
        fig = px.histogram(
            x=data,
            marginal='rug'
        )

        fig.update_layout(
            title=name,
        )

        fig.show()

    def decision_boundary(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model: BaseEstimator
    ):
        x0 = x[:, 0]
        x1 = x[:, 1]

        x_min, x_max = x0.min() - 1, x0.max() + 1
        y_min, y_max = x1.min() - 1, x1.max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )

        z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        z = z.astype(np.str)

        y = [str(label) for label in y]

        fig = px.scatter(x=x0, y=x1, color=y)

        # fig = go.Figure()

        contour = go.Contour(
            z=z,
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            line_width=0,
            colorscale = [
                [0, '#ff9900'],
                [1, '#6666ff']
            ],
            opacity=0.2,
            showscale=False
        )
        
        fig.add_trace(contour)

        fig.update_layout(
            title='Decision boundary',
            legend_title='Label'
        )

        fig.show()
