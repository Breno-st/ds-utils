from setuptools import find_packages, setup

setup(name='ensemble',
      version='0.1',
      description='Basic utilities for data science',
      url='https://github.com/breno-st/ds-utils/src',
      author='Breno Tiburcio',
      author_email='b_tiburcio@hotmail.com',
      license='MIT',
      packages=find_packages(where="ensemble"),
      package_dir={"": "ensemble"},
      zip_safe=False)


