from setuptools import setup, find_packages

setup(
    name='mon_projet',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'fastapi',
        'uvicorn',
        'streamlit',
        'pytest',
        'pyyaml'
    ],
    author="Louis POLART",
    description="Machine learning Rest API template project.",
)
