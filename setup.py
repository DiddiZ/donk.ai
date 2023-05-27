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
        "numpy~=1.23",
        "scipy~=1.10",
        "scikit-learn~=1.0",
        "matplotlib~=3.7",
        "seaborn~=0.12",
        "pandas~=1.5",
        "tensorflow~=2.12",
        "tensorflow-probability~=0.20",
        "keras~=2.12",
        "sympy~=1.12",
        "tqdm",
    ],
    project_urls={
        "Bug Reports": "https://github.com/DiddiZ/donk.ai/issues",
        "Source": "https://github.com/DiddiZ/donk.ai",
    },
)
