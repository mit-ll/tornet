from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name='tornet',
    version='1.0',
    description='Tools for working with TorNet dataset in python',
    url='https://github.com/mit-ll/tornet',
    author='MIT Lincoln Laboratory, Group 43',
    author_email='mark.veillette@ll.mit.edu',
    packages=find_packages(),
    package_data={},
    install_requires=[ 
    ],
    ext_modules=[] )