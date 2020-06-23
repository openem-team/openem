import setuptools

reqs=open('requirements.txt')
setuptools.setup(
    name='openem',
    version='0.2.0',
    url='https://github.com/openem-team/openem',
    author='CVision AI',
    author_email='info@cvisionai.com',
    maintainer='CVision AI',
    maintainer_email='info@cvisionai.com',
    packages=['openem', 'openem.Detect','openem.tracking'],
    # Don't require keras_retina net as that is only for
    # optional detector model, and not on pypi
    install_requires=reqs.readlines(),
    extra_requires=['tensorflow-gpu>=1.15.0,<2.0.0',
                    'tensorflow>=1.15.0,<2.0.0']
)
