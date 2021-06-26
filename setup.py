from setuptools import find_packages, setup

setup(name='ensemble',
      version='0.1',
      description='Basic utilities for data science',
      url='https://github.com/breno-st/ds-utils',
      author='Breno Tiburcio',
      author_email='b_tiburcio@hotmail.com',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)


setup(name='optimization',
      version='0.1',
      description='Basic utilities for data science',
      url='https://github.com/breno-st/ds-utils',
      author='Breno Tiburcio',
      author_email='b_tiburcio@hotmail.com',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)
