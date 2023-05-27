from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="donk.ai",
    version="0.1.0",
    description="Reinforcement Learning Toolbox",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    url="https://github.com/DiddiZ/donk.ai",
    author="Robin Kupper",
    author_email="robin.kupper@diddiz.de",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.8, <4",
    install_requires=[
        "numpy~=1.21",
        "scipy~=1.7.0",
        "scikit-learn~=0.24.2",
        "matplotlib~=3.4.2",
        "seaborn~=0.11.1",
        "pandas~=1.3.1",
        "tensorflow~=2.8.0",
        "tensorflow-probability~=0.15.0",
        "keras~=2.8.0",
        "sympy~=1.9",
        "tqdm",
    ],
    project_urls={
        "Bug Reports": "https://github.com/DiddiZ/donk.ai/issues",
        "Source": "https://github.com/DiddiZ/donk.ai",
    },
)
