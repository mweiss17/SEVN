from setuptools import setup

setup(
    name='SEVN_gym',
    version='1.1.1',
    install_requires=[
        'gym', 'pybullet>=1.9.4', 'scipy', 'pandas', 'tqdm',
        'matplotlib', 'numpy', 'networkx', 'pygame', 'h5py', 'tables',
        'dask[complete]', 'academictorrents>=2.3.2', 'enum34', 'numpy', 'comet_ml', 'opencv-python'
    ])
