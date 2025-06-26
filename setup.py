from setuptools import setup, find_packages

# Dependencies required for this project
dependencies = [
    'torch==2.6.0',
    'torchvision',
    'torchaudio',
    'transformers==4.46.3',
    'accelerate==1.0.1',
    'pydantic==2.9.2',
    'datasets==3.0.2',
    'peft==0.13.2',
    'fire',
    'clearml',
    'evaluate',
    'bitsandbytes',
    'lm-eval==0.4.8',
    'colorama',
    'pytorch-minimize',
    'termcolor',
]

setup(
    name="ReplaceMe",  # Update to your package's name
    version="0.1.1",
    packages=['ReplaceMe'],
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'get_distance_analysis=ReplaceMe.distance:run_from_config',
            'get_lt_with_lstsq=ReplaceMe.lstsq:run_from_config',
            'get_lt_with_solvers=ReplaceMe.cosine_dist:run_from_config',
            'evaluate_model=ReplaceMe.evaluator:run_from_config',
            'run_replaceme=ReplaceMe.ReplaceMe_pipeline:run_from_config',
            'run_uidl=ReplaceMe.UIDL_pipeline:run_from_config',
        ],
    },
    author="MTS AI Research Team",
    author_email="a.ammar@mts.ai",
    description="A highly efficient and robust package developed by MTS AI researchers.",
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/mts-ai/ReplaceMe",  # Update with your GitHub repository URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",  # Update with your project's license
    ],
    python_requires='>=3.10',
)
