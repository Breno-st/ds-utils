from setuptools import find_packages, setup

setup(name='ds-utils',
      version='0.1',
      description='Basic utilities for data science',
      url='https://github.com/breno-st/ds-utils',
      author='Brendan Hasz',
      author_email='b_tiburcio@hotmail.com',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)
