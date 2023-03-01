from typing import Union

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class Visualizer():
    """
    A class used to visualize features using PCA and TSNE.

    Attributes:
        `df` (pd.DataFrame): Dataframe with unique ids under
        `ID` column and targets under `Target` column.
        `X` (np.ndarray): Features got from input dataframe
        and scaled (optionally).
        `plotly_args` (dict): Dict with args for plotly charts.

    Methods:
        `set_plotly_args(**kwargs)`: Sets args for plotly charts.
        `pca2d()`: Uses PCA with 2 components and plots 2D chart.
        `pca3d()`: Uses PCA with 3 components and plots 3D chart.
        `tsne2d(perplexity, n_iter, pca_reduction)`: Uses TSNE
        and plots 2D chart.
        `tsne3d(perplexity, n_iter, pca_reduction)`: Uses TSNE
        and plots 3D chart.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        target_column: str,
        scale: bool = True
    ):
        """
        Args:
            `df` (pd.DataFrame): Dataframe with numerical features,
            unique ids and targets.
            `id_column` (str): Name of column with unique ids.
            `target_column` (str): Name of column with targets.
            `scale` (bool): Scale features by removing the mean and
            scaling to unit variance. Default is `True`.
        """
        self.df = pd.DataFrame(data={
            "ID": df[id_column],
            "Target": df[target_column]
        })
        self.X = df.drop(columns=["ID", "Target"]).values
        if scale:
            self.X = StandardScaler().fit_transform(self.X)
        self.set_plotly_args(font_size=14)

    def set_plotly_args(self, **kwargs):
        """
        Sets plotly args for charts.

        Args:
            `**kwargs`: named arguments for plotly `update_layout()` method
            (name of arguments must match arguments from this method).
            Example: `font_size=14, template='plotly_dark'`.
        """
        self.plotly_args = kwargs

    def _plot_2d(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
    ):
        fig = px.scatter(
            data_frame=data,
            x=x,
            y=y,
            custom_data=["ID", "Target"],
            hover_name="ID",
            hover_data=["Target"]
        )

        fig.update_traces(marker=dict(
            size=20,
            color=data["Target"],
            # colorscale='Inferno',
            showscale=True,
            line_width=1
        ),
            selector=dict(mode='markers'))

        fig.update_layout(**self.plotly_args)
        fig.show()

    def _plot_3d(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        z: str
    ):
        fig = px.scatter_3d(
            data_frame=data,
            x=x,
            y=y,
            z=z,
            custom_data=["ID", "Target"],
            hover_name="ID",
            hover_data=["Target"]
        )

        fig.update_traces(marker=dict(
            size=15,
            color=data["Target"],
            # colorscale='Inferno',
            showscale=True,
            line_width=2
        ),
            selector=dict(mode='markers'))

        fig.update_layout(**self.plotly_args)
        fig.show()

    def pca2d(self):
        """
        Uses PCA with 2 components and plots 2D chart.
        """
        pca = PCA(n_components=2, random_state=42)
        principal_components = pca.fit_transform(self.X)

        print("[INFO] Explained variaence ratio: "
              f"{pca.explained_variance_ratio_}")

        x_col, y_col = 'PCA_1', 'PCA_2'

        vis_df = pd.DataFrame(
            data=principal_components[:, 0:2],
            columns=[x_col, y_col]
        )
        vis_df = pd.concat([vis_df, self.df], axis=1)

        self._plot_2d(vis_df, x_col, y_col)

    def pca3d(self):
        """
        Uses PCA with 3 components and plots 3D chart.
        """
        pca = PCA(n_components=3, random_state=42)
        principal_components = pca.fit_transform(self.X)

        print("[INFO] Explained variaence ratio: "
              f"{pca.explained_variance_ratio_}")

        x_col, y_col, z_col = 'PCA_1', 'PCA_2', 'PCA_3'

        vis_df = pd.DataFrame(
            data=principal_components[:, 0:3],
            columns=[x_col, y_col, z_col]
        )
        vis_df = pd.concat([vis_df, self.df], axis=1)

        self._plot_3d(vis_df, x_col, y_col, z_col)

    def tsne2d(
        self,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        pca_reduction: Union[None, int] = None,
    ):
        """
        Uses TSNE and plots 2D chart.

        Args:
            `perplexity` (float): The perplexity is related to the number of
            nearest neighbors that is used in other manifold learning
            algorithms. Larger datasets usually require a larger perplexity.
            Consider selecting a value between 5 and 50. Different values
            can result in significantly different results. The perplexity
            must be less that the number of samples. Default is `30.0`.
            `n_iter` (int): Maximum number of iterations for the optimization.
            Should be at least 250. Default is `1000`.
            `pca_reduction` (int): Number of components for PCA to use
            before TSNE if specified. If `None` do not use PCA before TSNE.
            Default is `None`.
        """
        X = self.X

        if pca_reduction:
            pca = PCA(n_components=pca_reduction)
            X = pca.fit_transform(X)

        tsne = TSNE(
            random_state=47,
            n_components=2,
            verbose=0,
            perplexity=perplexity,
            n_iter=n_iter,
            n_jobs=-1
        ).fit_transform(X)

        x_col, y_col = 'TSNE_1', 'TSNE_2'

        vis_df = pd.DataFrame(
            data=tsne[:, 0:2],
            columns=[x_col, y_col]
        )
        vis_df = pd.concat([vis_df, self.df], axis=1)

        self._plot_2d(vis_df, x_col, y_col)

    def tsne3d(
        self,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        pca_reduction: Union[None, int] = None,
    ):
        """
        Uses TSNE and plots 3D chart.

        Args:
            `perplexity` (float): The perplexity is related to the number of
            nearest neighbors that is used in other manifold learning
            algorithms. Larger datasets usually require a larger perplexity.
            Consider selecting a value between 5 and 50. Different values
            can result in significantly different results. The perplexity
            must be less that the number of samples. Default is `30.0`.
            `n_iter` (int): Maximum number of iterations for the optimization.
            Should be at least 250. Default is `1000`.
            `pca_reduction` (int): Number of components for PCA to use
            before TSNE if specified. If `None` do not use PCA before TSNE.
            Default is `None`.
        """
        X = self.X

        if pca_reduction:
            pca = PCA(n_components=pca_reduction)
            X = pca.fit_transform(X)

        tsne = TSNE(
            random_state=42,
            n_components=3,
            verbose=0,
            perplexity=perplexity,
            n_iter=n_iter,
            n_jobs=-1
        ).fit_transform(X)

        x_col, y_col, z_col = 'TSNE_1', 'TSNE_2', 'TSNE_3'

        vis_df = pd.DataFrame(
            data=tsne[:, 0:3],
            columns=[x_col, y_col, z_col]
        )
        vis_df = pd.concat([vis_df, self.df], axis=1)

        self._plot_3d(vis_df, x_col, y_col, z_col)
