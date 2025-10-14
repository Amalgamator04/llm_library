from setuptools import setup, find_packages
import os
import io

# Utility function to read the contents of your README file
def read(*names, **kwargs):
    """Read contents of a file in the current directory."""
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

# Read the contents of README.md for the long description
long_description = read("README.md")

# Read the requirements from requirements.txt
requirements = read("requirements.txt").splitlines()

setup(
    name='llmconnections',
    version='1.0.0',
    author='Prabhav Sharma',
    author_email='prabhavs2004@gmail.com',
    description='A unified Python library for connecting to multiple LLM providers (OpenRouter, Perplexity, Hugging Face, SambaNova).',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='N.A.', 
    
    # This tells setuptools to find the 'llmconnections' folder and include it as the package
    packages=find_packages(), 
    
    # Automatically install the dependencies listed in requirements.txt
    install_requires=requirements,
    
    # Metadata for PyPI search and compatibility
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.8', 
)
