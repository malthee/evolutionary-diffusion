from setuptools import setup, find_packages

setup(
    name='evolutionary_model_helpers',
    version='0.1.0',
    author='malthee',
    description='Helper package for auto-loading different model types on devices. With additional utility functions.',
    packages=find_packages(),
    install_requires=[
        'torch~=2.1.0',
        'diffusers~=0.25.0',
        'transformers~=4.36.0',
        'Pillow>=10.1.0',
    ],
)
