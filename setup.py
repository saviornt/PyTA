from setuptools import setup, find_packages

setup(
    name="PyTA",
    version="0.1.0",
    description="PyTA is a Python library for technical analysis, offering a range of functions to compute indicators like moving averages, momentum, volatility, and patterns. Designed as a user-friendly alternative to TA-Lib, it leverages pandas and numpy for ease of use.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="David Wadsworth",
    author_email="",
    url='https://github.com/saviornt/PyTA',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pandas',
        'scipy'
    ],
    extras_require={
        'dev': [
            'pytest',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',
)
