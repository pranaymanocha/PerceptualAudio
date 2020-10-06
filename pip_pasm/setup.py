import setuptools

with open("README.md","r") as fh:
    long_description= fh.read()
    
setuptools.setup(name='pip_pasm',version='0.1',scripts=['__init__.py'],author='Pranay Manocha',author_email='pranaymnch@gmail.com',description=long_description,long_description_content_type="text/markdown",url="https://github.com/pranaymanocha/test_pasm",packages=setuptools.find_packages(),classifiers=["Programming Language :: Python :: 2", "LICENCE :: OSI Approved :: MIT Licence","Operating System :: OS Independent",],)