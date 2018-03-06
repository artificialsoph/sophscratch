from setuptools import find_packages, setup

setup(
    name='sophscratch',
    version='0.1.0',
    description='Building Python from scratch',
    long_description="",
    author='Sophie Searcy',
    author_email='s@soph.info',
    install_requires=[
          'graphviz',
      ],
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
