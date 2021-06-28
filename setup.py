import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

PACKAGE_NAME = 'ds_utils'
AUTHOR = 'Breno Tiburcio'
AUTHOR_EMAIL = 'b_tiburcio@hotmail.com'
URL = 'https://github.com/breno-st/ds_utils'
DOWNLOAD_URL = 'https://github.com/breno-st/ds_utils'



LICENSE = 'MIT'
VERSION = (HERE / "VERSION").read_text()
DESCRIPTION = 'A set of data tools in Python'
LONG_DESCRIPTION = (HERE / "DESCRIPTION.md").read_text()
LONG_DESC_TYPE = "text/markdown"


INSTALL_REQUIRES = ['numpy'
      , 'pandas>=0.23.4'
      , 'itertools'
      , 'scipy'
      , 'matplotlib'
      , 'scikit-learn'
      , 'scikit-plot>=0.3.7'
      , 'itertools'
      , 'collections'
]

CLASSIFIERS = [
      'Programming Language :: Python :: 3'
]

PYTHON_REQUIRES = '>=3.5'

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages(),
      classifiers=CLASSIFIERS
      )
