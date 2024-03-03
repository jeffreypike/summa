from setuptools import setup, find_packages

try:
    with open("README.md") as f:
        long_description = f.read()
except IOError:
    long_description = ""

try:
    with open("requirements.txt") as f:
        requirements = [x.strip() for x in f.read().splitlines() if x.strip()]
except IOError:
    requirements = []

setup(name='summa',
      install_requires=requirements,
      version='1.0',
      description='Additive Modelling Package',
      author='Jeffrey Pike  ',
      author_email='jeffreypike.ai@gmail.com',
      license="MIT",
      url='git@github.com:jeffreypike/summa.git',
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
      ],
      include_package_data=True
     )