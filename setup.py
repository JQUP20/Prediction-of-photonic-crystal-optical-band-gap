"""
Setup script for Photonic Crystal Band Gap Prediction package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="photonic-crystal-bandgap",
    version="0.1.0",
    author="Photonic Crystal Research Team",
    description="GNN-based prediction of photonic crystal optical band gaps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Prediction-of-photonic-crystal-optical-band-gap",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "photonic-train=train:main",
            "photonic-eval=evaluate:main",
        ],
    },
)
