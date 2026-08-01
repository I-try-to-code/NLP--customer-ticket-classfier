import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "customer-support-ai"
AUTHOR_USER_NAME = "Ayan"
SRC_REPO = "src"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="A small python package for NLP app",
    long_description=long_description,
    long_description_content="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
