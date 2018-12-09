from setuptools import setup, find_packages

setup(name='keras-adain-style-transfer',
      version=open("adain/_version.py").readlines()[-1].split()[-1].strip("\"'"),
      description='AdaIN style trainfer implementation in keras',
      author='jeongjoonsup',
      author_email='penny4860@gmail.com',
      # url='https://penny4860.github.io/',
      packages=find_packages(),
     )