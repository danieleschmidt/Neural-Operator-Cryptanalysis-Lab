#!/usr/bin/env python3
"""Setup script for Neural Operator Cryptanalysis Lab."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-operator-cryptanalysis",
    version="0.1.0",
    author="Terragon Labs",
    author_email="research@terragonlabs.com",
    description="Neural operators for defensive side-channel analysis of post-quantum cryptography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/Neural-Operator-Cryptanalysis-Lab",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "h5py>=3.7.0",
        "cryptography>=3.4.8",
        "pycryptodome>=3.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.10.0",
            "tensorboard>=2.10.0",
            "wandb>=0.13.0",
        ],
        "hardware": [
            "pyserial>=3.5",
            "pyvisa>=1.12.0",
            "picosdk>=1.1.0",
            "chipwhisperer>=5.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-sca=neural_cryptanalysis.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)