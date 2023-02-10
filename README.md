![example workflow](https://github.com/VolodymyrVozniak/feature-visualizer/actions/workflows/test.yml/badge.svg)

# Table of Contents
<ul>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#dependencies">Dependencies</a></li>
  <li><a href="#tutorials">Tutorials</a></li>
  <li><a href="#usage">Usage</a></li>
</ul>

</br>

# Installation

To install this repo as Python lib just run the following command:

```sh
pip install git+https://github.com/VolodymyrVozniak/feature-visualizer
```

</br>

# Dependencies

```sh
scikit-learn>=1.2.0
pandas>=1.5.2
plotly>=5.7.0
```

</br>

# Tutorials

1. For binary problem check [this tutorial](https://colab.research.google.com/drive/12jAykZuKHp3YBA56kxaVP-h7E9K4Lzw3)
2. For regression problem check [this tutorial](https://colab.research.google.com/drive/19rEMcOVogDBujbIbtqU_PJr5QJs2UCw2)
3. For multiclassification problem check [this tutorial](https://colab.research.google.com/drive/1C0ZwHfTir4WuLStev-VAZMA5ftSYq2s-)

</br>

# Usage

There is 1 main class for feature visualization:
* `Visualizer` - to visualize features for binary, regression and multiclassification problems

</br>

## Examples

* Binary problem

```python
from sklearn.datasets import load_breast_cancer
from croatoan_visualizer import Visualizer


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
```

For more details check [tutorial](https://colab.research.google.com/drive/12jAykZuKHp3YBA56kxaVP-h7E9K4Lzw3)

<p align="right">(<a href="#top">back to top</a>)</p>

* Regression problem

```python
from sklearn.datasets import load_diabetes
from croatoan_visualizer import Visualizer


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
```

For more details check [tutorial](https://colab.research.google.com/drive/19rEMcOVogDBujbIbtqU_PJr5QJs2UCw2)

<p align="right">(<a href="#top">back to top</a>)</p>

* Multiclassification problem

```python
from sklearn.datasets import load_iris
from croatoan_visualizer import Visualizer


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
```

For more details check [tutorial](https://colab.research.google.com/drive/1C0ZwHfTir4WuLStev-VAZMA5ftSYq2s-)

<p align="right">(<a href="#top">back to top</a>)</p>
