from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nemus",
    version="1.0.0",
    author="Charan",
    author_email="yellapragadacharankrishna1234@gmail.com",
    description="Event-driven neuromorphic computing library with analytical precision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ycharankrishna/NEMUS",
    project_urls={
        "Bug Tracker": "https://github.com/ycharankrishna/NEMUS/issues",
        "Documentation": "https://nemus.readthedocs.io",
        "Source Code": "https://github.com/ycharankrishna/NEMUS",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "viz": [
            "matplotlib>=3.3.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
    },
    keywords=[
        "neuromorphic", 
        "spiking-neural-networks", 
        "event-driven", 
        "spike-timing",
        "neuromorphic-computing",
        "loihi",
        "analog-computing",
        "brain-inspired",
    ],
    include_package_data=True,
    zip_safe=False,
)
