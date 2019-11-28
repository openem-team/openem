import setuptools

setuptools.setup(
    name='openem',
    version='0.2.0',
    url='https://github.com/openem-team/openem',
    author='CVision AI',
    author_email='info@cvisionai.com',
    maintainer='CVision AI',
    maintainer_email='info@cvisionai.com',
    packages=['openem', 'openem.Detect'],
    # Don't require keras_retina net as that is only for
    # optional detector model, and not on pypi
    install_requires=['tensorflow-gpu>=1.14.0,<2.0.0', 'progressbar2>=3.43.1']
)
