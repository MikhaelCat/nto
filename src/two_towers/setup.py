# setup.py в корне проекта
from setuptools import setup, find_packages

setup(
    name="two_towers",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "tqdm>=4.64.0",
        "transformers>=4.25.0",
        "sentence-transformers>=2.2.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyarrow>=10.0.0",
    ],
)