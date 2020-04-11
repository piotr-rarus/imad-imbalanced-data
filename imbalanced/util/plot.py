# flake8: noqa

from typing import Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from sklearn.decomposition import PCA


class Plot:

    def __init__(
        self,
        image_width: int = 1200,
        image_height: int = 900,
        debug_mode: bool = False
    ) -> None:

        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.DEBUG_MODE = debug_mode

        if not self.DEBUG_MODE:
            pyo.init_notebook_mode(connected=False)
            pio.renderers.default = 'notebook'

    def class_balance(
        self,
        class_balance: Dict[str, int],
        filename='class_balance'
    ):
        """
        Plots class balance.
        
        Parameters
        ----------
        class_balance : Dict[str, int]
            Record count for each label by name.
        filename : str, optional
            Plotly artifact.
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

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )

    # TODO 3d scatter plot
    def features_distribution(
        self,
        x: np.ndarray,
        y: np.ndarray,
        filename='features_distribution'
    ):
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

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )


    def heatmap(
        self,
        data: np.ndarray,
        plot_name: str,
        ylabel='',
        xlabel='',
        filename=''
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

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )

    def scores(self, scores: Dict[str, List[float]], filename: str = ''):
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

    def boxplot(self, data: Dict[str, List[float]], filename: str = ''):

        fig = go.Figure()

        for metric, values in data.items():

            fig.add_trace(
                go.Box(
                    y=values,
                    name=metric
                )
            )

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )

    def distribution(self, data: List[float], name: str='', filename: str = ''):
        fig = px.histogram(
            x=data,
            marginal='rug'
        )

        fig.update_layout(
            title=name,
        )

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )
