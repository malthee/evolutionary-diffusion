from setuptools import setup, find_packages

setup(
    name='evolutionary_prompt_embedding',
    version='0.1.0',
    author='malthee',
    description='Generating images using prompt embeddings and evolutionary algorithms. '
                'Part of the evolutionary suite.',
    packages=find_packages(),
    package_data={
        'evolutionary_prompt_embedding': ['tensors/*.pt'],
    },
    include_package_data=True,
    install_requires=[
        'evolutionary~=0.1.0',
        'evolutionary_imaging~=0.1.0',
        'datasets~=2.16.0',
    ],
)
