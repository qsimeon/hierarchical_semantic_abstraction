from setuptools import setup, find_packages

setup(
    name="hierarchical_semantic_abstraction",
    version="0.1.0",
    description="A Multi-Modal Framework for Controlled Image Generation with Hierarchical Semantic Abstraction",
    author="[Your Name]",
    author_email="[Your Email]",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#") and not line.startswith("git+")
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)