import pandas as pd
from sklearn.datasets import load_diabetes

from croatoan_visualizer import Visualizer


def test_regression():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    df = X.assign(target=y)
    df["target"] = pd.qcut(df["target"], 10, labels=False, duplicates='drop')

    vis = Visualizer(
        df=df.reset_index(),
        id_column="index",
        target_column="target"
    )

    vis.pca2d()
    vis.pca3d()

    vis.tsne2d()
    vis.tsne3d()

    vis.tsne2d(
        perplexity=20,
        n_iter=2000,
        pca_reduction=5
    )
    vis.tsne3d(
        perplexity=20,
        n_iter=2000,
        pca_reduction=5
    )
