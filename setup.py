import setuptools

import croatoan_visualizer


setuptools.setup(
    name='croatoan-visualizer',
    version=croatoan_visualizer.__version__,
    packages=setuptools.find_packages(),
    url='https://github.com/VolodymyrVozniak/universal-trainer',
    license='',
    author='Volodymyr Vozniak',
    author_email='vozniak.v.z@gmail.com',
    description='Feature visualizer',
    install_requires=[
        'scikit-learn>=1.2.0',
        'pandas>=1.5.2',
        'plotly>=5.7.0'
    ],
    python_requires=">=3.8",
)
