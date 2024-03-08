from setuptools import setup, find_packages

torch_diffusers_requirements = [
    'torch~=2.2.0',
    'diffusers~=0.26.0',
    'transformers~=4.38.0',
    'accelerate~=0.27.0',
    'Pillow',
]

extras_require = {
    'imaging': [
        'imageio',
        'numpy',
        # Used in the Aesthetics model for evaluation
        'clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33',
        'torchvision~=0.17.0',
        'torchmetrics~=1.3.0',
        'pytorch-lightning~=2.2.0',
        'imageio>=2.33.0',
    ] + torch_diffusers_requirements,
    'model_helpers': torch_diffusers_requirements,
    'prompt_embedding': torch_diffusers_requirements,
    'prompt_embedding_utils': [
        'datasets~=2.16.0',
    ] + torch_diffusers_requirements,
}

all_deps = set(dep for deps in extras_require.values() for dep in deps)
extras_require['all'] = list(all_deps)

setup(
    name='evolutionary',
    version='0.2.3',
    author='malthee',
    url='https://github.com/malthee/evolutionary-diffusion',
    description='''Base package defining a framework for evolutionary algorithms to be used with generative networks.
                   Splits up the Solution-Representation into arguments and results.''',
    long_description_content_type='text/markdown',
    long_description='''This package includes generic classes for evolutionary computation in a generational environment.
                     Crossover and Mutation happens on the argument (A) level, whilst the fitness is evaluated on the result (R) level.
                     SolutionCandidates are created by a SolutionCreator, their representation is split into arguments (A) and result (R).
                     
                     Additional installation variants (pip install evolutionary[...]):
                     * all: Install all dependencies, full functionality across all subpackages.
                     * imaging (evolutionary_imaging): Contains base dependencies for evolutionary image generation, 
                     evaluation and visualization.
                     * model_helpers (evolutionary_model_helpers): Auto-loading different model types on devices. 
                     With additional utility functions.
                     * prompt_embedding (evolutionary_prompt_embedding): Using evolutionary_prompt_embeddings to 
                     generate images and perform evolutionary variation using prompt embeddings.
                     * prompt_embedding_utils (evolutionary_prompt_embedding.utils): Additional utilities for evaluating
                     prompt embedding range. 
                     ''',
    packages=find_packages(),
    install_requires=[
        'tqdm>=4.66.0',  # For visualizing progress of algorithms
        'matplotlib>=3.7',
    ],
    package_data={
        'evolutionary_prompt_embedding': ['tensors/*'],
    },
    include_package_data=True,
    extras_require=extras_require,
)
