from setuptools import setup, find_packages

setup(
    name='KLAY',
    version='0.6.1',
    author='Amit Gupta',
    author_email='gupta839@umn.edu', 
    description='A Python package for generating ML layers for MLIPs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ipcamit/KLAY',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Minimum Python version required
    install_requires=[
        'torch',
        'torch_runstats',
        'e3nn',
    ],
    include_package_data=True,  # Include non-code files listed in MANIFEST.in
)
