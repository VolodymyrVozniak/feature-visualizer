from sklearn.datasets import load_breast_cancer

from croatoan_visualizer.visualizer import Visualizer


def test_binary():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
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
        pca_reduction=15,
        metric='cosine'
    )
    vis.tsne3d(
        perplexity=20,
        n_iter=2000,
        pca_reduction=15,
        metric='cosine'
    )
