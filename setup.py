from setuptools import setup, find_packages

setup(
    name='rustat-python-api',
    version='0.1.1',
    description='A Python wrapper for RuStat API',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Daniel Zholkovsky',
    author_email='daniel@zholkovsky.com',
    url='https://github.com/dailydaniel/rustat-python-api',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'requests==2.32.3',
        'pandas==2.2.3',
        'tqdm==4.66.5'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
