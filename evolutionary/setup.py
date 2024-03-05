from setuptools import setup, find_packages

setup(
    name='evolutionary',
    version='0.1.2',
    author='malthee',
    description='''Base package defining a framework for evolutionary algorithms to be used with generative networks.
                   Splits up the Solution-Representation into arguments and results.''',
    long_description='''This package includes generic classes for evolutionary computation in a generational environment.
                     Crossover and Mutation happens on the argument (A) level, whilst the fitness is evaluated on the result (R) level.
                     SolutionCandidates are created by a SolutionCreator, their representation is split into arguments (A) and result (R).''',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'tqdm'
    ],
)
