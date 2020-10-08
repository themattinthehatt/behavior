from distutils.core import setup

setup(
    name='behavior',
    version='0.0.0',
    description='a collection of tools for analyzing behavioral data',
    author='matt whiteway',
    author_email='',
    url='http://www.github.com/themattinthehatt/behavior',
    install_requires=[
        'numpy', 'matplotlib', 'sklearn', 'scipy==1.1.0', 'jupyter', 'seaborn'],
    packages=['behavior'],
)
