from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
        "tqdm>=4.39.0",
        "dill>=0.3.2",
        "cloudml-hypertune>=0.1.0.dev6"
        ]

setup(
    name='sensor-ml-training',
    author='Jorge III Altamirano-Astorga',
    author_email='jorge3a(at)gmail.com',
    url='https://philwebsurfer.github.io/dlfinal/mlmodels-src/',
    version='0.23',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Training application for the Machine Learning Models in "Air Quality Sensor" deep learning research.'
)
