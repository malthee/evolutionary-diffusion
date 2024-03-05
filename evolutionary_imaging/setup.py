from setuptools import setup, find_packages

setup(
    name='evolutionary_imaging',
    version='0.1.0',
    author='malthee',
    description='Base package for evolutionary image generation and evaluation. To be used with the evolutionary '
                'package.',
    packages=find_packages(),
    install_requires=[
        'evolutionary~=0.1.0',
        'model_helpers~=0.1.0',
        'pillow',
        'imageio',
        'numpy',
        # Used in the Aesthetics model
        'clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'
        'torch~=2.1.0',
        'torchvision~=0.16.0',
        'torchmetrics~=1.2.0',
        'diffusers~=0.25.0',
        'pytorch-lightning~=2.1.0',
        'transformers~=4.36.0',
    ],
)
