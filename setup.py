from setuptools import setup, find_packages

setup(
    name='dataset_builder',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-learn',
    ],
    author='Vinicius Costa',
    description='Organizador automático de datasets com imagens e labels',
    license='MIT',
)
