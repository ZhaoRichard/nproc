from setuptools import setup

setup(
   name='nproc',
   version='1.0',
   description='Neyman-Pearson (NP) Classification Algorithms and NP Receiver Operating Characteristic (NP-ROC) Curves',
   author='Richard Zhao, Yang Feng, Jingyi Jessica Li and Xin Tong',
   author_email='zhao.rich@gmail.com',
   packages=['nproc'],
   install_requires=['scipy', 'sklearn', 'numpy', 'joblib', 'multiprocessing'],
   long_description=open('README.md').read(),
)