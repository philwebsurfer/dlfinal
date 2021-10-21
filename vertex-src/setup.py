from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
        "matplotlib>=3.1.2",
        "tqdm>=4.39.0",
        "dill>=0.3.2",
        "h5py>=2.10.0"
        ]

setup(
    name='sensor-training',
    author='Jorge III Altamirano-Astorga',
    author_email='jorge3a(at)gmail.com',
    url='https://philwebsurfer.github.io/dlfinal/vertex-src/',
    version='0.5',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Training application for the "Air Quality Sensor" deep learning research.'
)
