import pandas as pd
from croatoan_visualizer import Visualizer


df = pd.read_csv('data/credit_risk.csv').reset_index()

vis = Visualizer(df, "index", "loan_status")

vis.pca2d()
vis.pca3d()
vis.tsne2d(perplexity=30, n_iter=1000, pca_reduction=10)
vis.tsne3d(perplexity=30, n_iter=1000, pca_reduction=10)
