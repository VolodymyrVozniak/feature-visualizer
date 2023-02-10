from sklearn.datasets import load_iris

from croatoan_visualizer.visualizer import Visualizer


def test_multiclass():
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = X.assign(target=y)

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
        pca_reduction=3
    )
    vis.tsne3d(
        perplexity=20,
        n_iter=2000,
        pca_reduction=3
    )
