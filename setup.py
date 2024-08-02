from setuptools import setup, find_packages

setup(name='pympc',
      version='0.1',
      description='Toolbox for Model Predictive Control of linear and hybrid systems',
      url='https://github.com/TobiaMarcucci/pympc',
      author='Tobia Marcucci',
      author_email='tobiam@mit.edu',
      license='MIT',
      packages=find_packages(),
      keywords=[
          'model predictive control',
          'computational geometry'
          ],
      install_requires=[
          'six',
          'numpy',
          'scipy<=1.11.4',
          'matplotlib',
          'gurobipy'
      ],
      zip_safe=False)
