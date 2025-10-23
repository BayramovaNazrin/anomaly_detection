from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1.0",
    author="Nazrin Bayramova",
    author_email="your.email@example.com",
    description="Anomaly detection for illicit Bitcoin transactions using GraphSAGE and RandomForest models.",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "matplotlib",
        "networkx",
        "tqdm"
    ],
    python_requires=">=3.8",
)
