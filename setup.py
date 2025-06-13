"""
Setup configuration for Transformer-Squared Framework with Gemma 3n
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("transformer_squared/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="transformer-squared-gemma3n",
    version="1.0.0",
    author="Transformer-Squared Team",
    author_email="contact@transformer-squared.ai",
    description="Self-adaptive LLM framework with Google's Gemma 3n E4B model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/transformer-squared/gemma3n",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "transformer-squared-demo=transformer_squared.examples.basic_usage:main",
            "transformer-squared-test=test_framework:main",
        ],
    },
    include_package_data=True,
    package_data={
        "transformer_squared": [
            "requirements.txt",
            "examples/*.py",
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "transformer",
        "gemma",
        "gemma-3n",
        "self-adaptive",
        "llm",
        "machine-learning",
        "deep-learning",
        "nlp",
        "artificial-intelligence",
        "parameter-efficient",
        "fine-tuning",
        "expert-systems",
    ],
    project_urls={
        "Bug Reports": "https://github.com/transformer-squared/gemma3n/issues",
        "Source": "https://github.com/transformer-squared/gemma3n",
        "Documentation": "https://transformer-squared.readthedocs.io/",
        "Research Paper": "https://arxiv.org/html/2501.06252v3",
        "Gemma 3n Model": "https://huggingface.co/google/gemma-3n-E4B-it-litert-preview",
    },
) 