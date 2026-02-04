from setuptools import setup, find_packages

setup(
    name="dgadb",
    version="0.1.0",
    description="Dynamic Graph Anomaly Detection Benchmark",
    author="DGADB Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.0.0",
        "polars>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "networkx>=3.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
    ],
)
