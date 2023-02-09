import pandas as pd
from croatoan_visualizer import Visualizer


df = pd.read_csv('data/credit_risk.csv')
df = df.reset_index().rename(columns={'index': 'id', 'loan_status': 'target'})

vis = Visualizer(df)

vis.pca2d()

# vis.pca3d()

# vis.tsne2d(perplexity=30, n_iter=2500, pca_reduction=False, color='target')

# vis.tsne3d(perplexity=30, n_iter=2500, pca_reduction=False, color='target')
