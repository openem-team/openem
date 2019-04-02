# Contributing

We welcome contributions to OpenEM! Please follow our guidelines below for issues and pull requests. We also have a section on adding new algorithm models to OpenEM.

Before starting work on OpenEM, feel free to contact the project maintainers at info@cvisionai.com. We are happy to discuss your ideas and help you define a path forward.

## Pull Requests

1. Make sure your code conforms to the [Google style guide][GoogleCpp] for C++. Python code should be checked using [pylint][pylint] and each file should have a score of at least 9.0.
2. Before initiating a pull request, please make sure you can still run both the top level train and test scripts for the relevant portions of code modified.
3. If you are modifying the inference scripts, please ensure the python and C# bindings still work.
4. Pull requests will be accepted when two project members have reviewed and approved it.

## Issues

Issues should include the following:

1. A brief summary of the issue.
2. Whether the issue occurs in the docker image or Windows native.
3. Steps to reproduce the issue.

## Adding algorithm models

One modification we expect users may want to make is to try a different network architecture. Our architectures are defined in the following files:

```shell
train
|-- openem_train
|   |-- inception
|   |   |-- inception.py
|   |-- rnn
|   |   |-- rnn.py
|   |-- ssd
|   |   |-- ssd.py
|   |-- unet
|   |   |-- unet.py
```

A function at the top of each of these files (named rnn_model, inception_model, etc.) defines the model architecture. You can try modifying these functions to improve performance. If you find an architecture that consistently outperforms the original, let us know or create a pull request.

[GoogleCpp]: https://google.github.io/styleguide/cppguide.html
[pylint]: https://www.pylint.org/

