from setuptools import setup

setup(
   name='nproc',
   version='1.4.0',
   description='Neyman-Pearson (NP) Classification Algorithms and NP Receiver Operating Characteristic (NP-ROC) Curves',
   author='Richard Zhao, Yang Feng, Jingyi Jessica Li and Xin Tong',
   author_email='zhao.rich@gmail.com',
   url='https://github.com/ZhaoRichard/nproc',
   packages=['nproc'],
   install_requires=['scipy', 'sklearn', 'numpy', 'joblib'],
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
)