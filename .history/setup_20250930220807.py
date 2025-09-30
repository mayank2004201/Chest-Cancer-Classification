import setuptools

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Chest-Cancer-Classification-"
AUTHOR_USER_NAME = "mayank2004201"
SRC_REPO = "Chest Cancer Classification"
AUTHOR_EMAIL = "mayankgoel201@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    descrition = "A small python package for CNN app.",
    long_description=long_description,
    long_description_content="text/markdown",
    url  =f"https://github/{A}"
    
)