from typing import List

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class Visualizer():
    def __init__(self, df, id_col='id', target_col="target"):
        self.df = df
        self.id = df[id_col]
        self.y = df[target_col]
        self.X = StandardScaler().fit_transform(df.drop(columns=[id_col, target_col]))

    def plot_2d(self,
                data: pd.DataFrame,
                x: str,
                y: str,
                color: str = 'target',
                custom_data: List[str] = ['id', 'target'],
                hover_name: str = 'id',
                hover_data: List[str] = ['target'],
                width=960,
                height=640,
                colorscale='Inferno'):

        fig = px.scatter(data_frame=data,
                         x=x,
                         y=y,
                         width=width,
                         height=height,
                         custom_data=custom_data,
                         hover_name=hover_name,
                         hover_data=hover_data,
                         template='plotly_dark')

        fig.update_traces(marker=dict(
            size=20,
            color=data[color],
            colorscale=colorscale,
            showscale=True,
            line_width=1
        ),
            selector=dict(mode='markers'))

        return fig

    def plot_3d(self,
                data: pd.DataFrame,
                x: str,
                y: str,
                z: str,
                color: str = 'target',
                custom_data: List[str] = ['id', 'target'],
                hover_name: str = 'id',
                hover_data: List[str] = ['target'],
                width=960,
                height=640,
                colorscale='Inferno'):

        fig = px.scatter_3d(data_frame=data,
                            x=x,
                            y=y,
                            z=z,
                            width=width,
                            height=height,
                            custom_data=custom_data,
                            hover_name=hover_name,
                            hover_data=hover_data,
                            template='plotly_dark')

        fig.update_traces(marker=dict(
            size=20,
            color=data[color],
            colorscale=colorscale,
            showscale=True,
            line_width=1
        ),
            selector=dict(mode='markers'))

        return fig

    def _plot_2d(self, component1, component2, colors, width=960, height=640, colorscale='Inferno'): #portland, rdylbu
        fig = go.Figure(data=go.Scatter(
            x=component1,
            y=component2,
            mode='markers',
            marker=dict(
                size=20,
                color=colors,  # set color equal to a variable
                colorscale=colorscale,  # one of plotly colorscales
                showscale=True,
                line_width=1
            ),
            # hover_name=self.smiles,
            # hover_data=self.y
        ))
        fig.update_layout(margin=dict(l=100, r=100, b=100, t=100), width=width, height=height)
        fig.layout.template = 'plotly_dark'

        # fig.show()
        return fig


    def _plot_3d(self, component1, component2, component3, colors, width=960, height=640, colorscale='Inferno'):
        fig = go.Figure(data=[go.Scatter3d(
            x=component1,
            y=component2,
            z=component3,
            mode='markers',
            marker=dict(
                size=10,
                color=colors,  # set color equal to a variable
                colorscale=colorscale,  # one of plotly colorscales
                opacity=1,
                showscale=True,
                line_width=1
            )
        )])
        # tight layout
        fig.update_layout(margin=dict(l=50, r=50, b=50, t=50), width=width, height=height)
        fig.layout.template = 'plotly_dark'

        # fig.show()
        return fig


    def _pca2d(self, colors=None, **kwargs):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.X)

        print(f'Explained_var: {pca.explained_variance_ratio_}')

        if colors is None:
            colors = self.y

        return self.plot_2d(principalComponents[:, 0], principalComponents[:, 1], colors, **kwargs)

    def pca2d(self, color: str = 'target', **kwargs):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.X)

        print(f'Explained_var: {pca.explained_variance_ratio_}')

        x_col, y_col = 'pca1', 'pca2'

        vis_df = pd.DataFrame(principal_components[:, 0:2], columns=[x_col, y_col])
        vis_df = vis_df.assign(target=self.df['target'], id=self.df['id'])
        if 'hover_data' in kwargs.keys():
            for el in kwargs['hover_data']:
                vis_df = vis_df.assign(**{el:self.df[el]})

        return self.plot_2d(vis_df, x_col, y_col, color, **kwargs)

    def _pca3d(self, colors=None, **kwargs):
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(self.X)

        print(f'Explained_var: {pca.explained_variance_ratio_}')

        if colors is None:
            colors = self.y

        return self.plot_3d(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], colors, **kwargs)

    def pca3d(self, color: str = 'target', **kwargs):
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(self.X)

        print(f'Explained_var: {pca.explained_variance_ratio_}')

        x_col, y_col, z_col = 'pca1', 'pca2', 'pca3'

        vis_df = pd.DataFrame(principal_components[:, 0:3], columns=[x_col, y_col, z_col])
        vis_df = vis_df.assign(target=self.df['target'], id=self.df['id'])

        return self.plot_3d(vis_df, x_col, y_col, z_col, color, **kwargs)

    def tsne2d(self, perplexity=30, n_iter=1000, pca_reduction=False, color: str = 'target', **kwargs):
        X = self.X

        if pca_reduction:
            pca = PCA(n_components=pca_reduction)
            X = pca.fit_transform(X)

        tsne = TSNE(random_state=47, n_components=2, verbose=0, perplexity=perplexity, n_iter=n_iter, n_jobs=-1)\
            .fit_transform(X)

        x_col, y_col = 'x', 'y'

        vis_df = pd.DataFrame(tsne[:, 0:2], columns=[x_col, y_col])
        vis_df = vis_df.assign(target=self.df['target'], id=self.df['id'])

        return self.plot_2d(vis_df, x_col, y_col, color, **kwargs)


    def _tsne3d(self, perplexity=30, n_iter=1000, pca_reduction=False, colors=None, **kwargs):
        X = self.X

        if pca_reduction:
            pca = PCA(n_components=pca_reduction)
            X = pca.fit_transform(X)

        tsne = TSNE(random_state=42, n_components=3, verbose=0, perplexity=perplexity, n_iter=n_iter) \
            .fit_transform(X)

        if colors is None:
            colors = self.y

        return self.plot_3d(tsne[:, 0], tsne[:, 1], tsne[:, 2], colors, **kwargs)


    def tsne3d(self, perplexity=30, n_iter=1000, pca_reduction=False, color: str = 'target', **kwargs):
        X = self.X

        if pca_reduction:
            pca = PCA(n_components=pca_reduction)
            X = pca.fit_transform(X)

        tsne = TSNE(random_state=42, n_components=3, verbose=0, perplexity=perplexity, n_iter=n_iter, n_jobs=-1) \
            .fit_transform(X)

        x_col, y_col, z_col = 'pca1', 'pca2', 'pca3'

        vis_df = pd.DataFrame(tsne[:, 0:3], columns=[x_col, y_col, z_col])
        vis_df = vis_df.assign(target=self.df['target'], id=self.df['id'])

        return self.plot_3d(vis_df, x_col, y_col, z_col, color, **kwargs)

