import setuptools

reqs=open('requirements.txt')
setuptools.setup(
    name='openem2',
    version='0.1.0',
    url='https://github.com/openem-team/openem',
    author='CVision AI',
    author_email='info@cvisionai.com',
    maintainer='CVision AI',
    maintainer_email='info@cvisionai.com',
    packages=['openem2'],
    # Don't require keras_retina net as that is only for
    # optional detector model, and not on pypi
    install_requires=reqs.readlines(),
    extra_requires=['tensorflow-gpu>=2.2.0,<3.0.0',
                    'tensorflow-hub',
                    'tensorflow>=2.2.0,<3.0.0']
)
