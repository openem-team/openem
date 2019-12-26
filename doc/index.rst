.. pytator documentation master file, created by
   sphinx-quickstart on Sun Dec  8 00:18:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenEM: Open Source Electronic Monitoring Library
=================================================

**OpenEM** is a library that provides advanced video analytics for
fisheries electronic monitoring (EM) data.  It currently supports detection,
classification, counting and measurement of fish during landing or discard.
This functionality is available via a deployment library with
pretrained models available in our example data (see tutorial).  The base
library is written in C++, with bindings available for both Python and C#.
Examples are included for all three languages.

The current release also includes a training library for the all OpenEM
functionality. The library is distributed as a native Windows library and
as a Docker image.

Click the image below to see a video of OpenEM in action:

`OpenEM detection example <https://youtu.be/EZ1Xyg_mnhM>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   FAQS
   tutorial
   api/deployment
   data_collection
   annotation
   training_environment
   build
   build/docker/build.md

Change Log
============

+---------+-----------+-------------------------------+
| Version | Date      | Description of changes        |
+=========+===========+===============================+
|         |           | - Add RetinaNet based         |
|         |           |   detector                    |
|         |           | - Deprecate C++ library       |
| 0.1.3   | Jan-2020  | - Add python inference        |
+---------+-----------+-------------------------------+
|         |           | - Docker image cleanups       |
|         |           | - Add Improved documentation  |
| 0.1.2   | June-2019 |                               |
+---------+-----------+-------------------------------+
|         |           | - Fix training in Docker      |
|         |           |                               |
| 0.1.1   | Feb-2019  |                               |
+---------+-----------+-------------------------------+
|         |           | - First stable release        |
|         |           | - Training/Inference examples |
| 0.1.0   | Jan-2019  |                               |
+---------+-----------+-------------------------------+

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
