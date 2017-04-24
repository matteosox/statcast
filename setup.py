from setuptools import setup, find_packages


setup(name='statcast',
      version='1.0.0',
      description='A baseball Python project',
      long_description='Collecting, storing, manipulating, and visualizing '
      'baseball data, mostly from statcast.',
      url='https://github.com/matteosox/statcast',
      author='Matt Fay',
      author_email='matt.e.fay@gmail.com',
      classifiers=['Programming Language :: Python :: 3.5',
                   'Development Status :: 2 - Pre-Alpha',
                   'Natural Language :: English'],
      keywords='baseball statcast mlb sabermetrics',
      packages=find_packages(),
      package_data={'statcast': ['data/*.*', 'data/logos/*.png']})
