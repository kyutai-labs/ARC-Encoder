import io
import os
from typing import List

import setuptools

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="embed_llm",
    version="0.0",
    author="Hippolyte Pilchen, Edouard Grave, Patrick Pérez",
    license="Apache 2.0",
    description=("Embedding augmented LLM experiments"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        # "License :: OSI Approved :: Apache Software License",
        # "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # exclude=("benchmarks", "docs", "examples", "tests")),
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=get_requirements(),
)
