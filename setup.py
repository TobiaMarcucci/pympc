from setuptools import setup

setup(name='pympc',
      version='0.1',
      description='Toolbox for Model Predictive Control of linear and hybrid systems',
      url='https://github.com/TobiaMarcucci/pympc',
      author='Tobia Marcucci',
      author_email='tobiam@mit.edu',
      license='MIT',
      packages=['pympc'],
      keywords=[
          'model predictive control',
          'computational geometry'
          ],
      install_requires=[
          'six',
          'numpy',
          'scipy',
          'matplotlib'
      ],
      zip_safe=False)
