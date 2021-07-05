import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

PACKAGE_NAME = 'data_st'
AUTHOR = 'Breno Tiburcio'
AUTHOR_EMAIL = 'b_tiburcio@hotmail.com'
URL = 'https://github.com/breno-st/ds_utils'
DOWNLOAD_URL = 'https://github.com/breno-st/ds_utils'



LICENSE = 'MIT'
VERSION = (HERE / "VERSION").read_text()
DESCRIPTION = 'Some ultilities for data science in Python'
LONG_DESCRIPTION = (HERE / "DESCRIPTION.md").read_text()
LONG_DESC_TYPE = "text/markdown"


INSTALL_REQUIRES = ['numpy'
      , 'pandas>=0.23.4'
      , 'scipy'
      , 'matplotlib'
      , 'scikit-learn'
      , 'scikit-plot>=0.3.7'
      , 'more-itertools'
      , 'collections'

]

CLASSIFIERS = [
      'Programming Language :: Python :: 3'
]

PYTHON_REQUIRES = '>=3.5'

setup(name=PACKAGE_NAME,
      version=VERSION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      url=URL,
      classifiers=CLASSIFIERS,
      license=LICENSE,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      #packages=find_packages(),
      package_dir={"": "src"},
      packages=find_packages(where="src")

      )
