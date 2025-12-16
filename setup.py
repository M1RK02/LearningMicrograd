from setuptools import setup, find_packages

setup(
    name='micrograd',
    version='0.1.0',
    description='A tiny scalar-valued autograd engine learning project',
    author='Mirko Pica',
    author_email='picamirko02@gmail.com',
    url='https://github.com/M1RK02/LearningMicrograd',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
