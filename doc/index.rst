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

Watch the video below to see a video of OpenEM in action:

.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/EZ1Xyg_mnhM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   data_collection
   annotation
   build
   build/docker/build.md
   FAQS
   api/deployment
   pipelines
   tracking
   classification

.. toctree::
   :maxdepth: 2
   :caption: Legacy Documentation (<=0.1.2):

   legacy_tutorial
   build/docker/legacy_build.md

Change Log
============

+---------+-----------+---------------------------------------------------------------------+
| Version | Date      | Description of changes                                              |
+=========+===========+=====================================================================+
|         |           | - Add support and scripts for Tator support                         |
|         |           | - Make image models multi-process capable                           |
|         |           | - Documentation improvements                                        |
| 0.1.4   | XXX-2020  |                                                                     |
+---------+-----------+---------------------------------------------------------------------+
|         |           | - Add RetinaNet based detector                                      |
|         |           | - Add pure python inference library                                 |
|         |           | - Deprecate C++ library                                             |
| 0.1.3   | Jan-2020  | - Alpha support for Xavier-based platforms                          |
+---------+-----------+---------------------------------------------------------------------+
|         |           | - Docker image cleanups                                             |
|         |           | - Add Improved documentation                                        |
| 0.1.2   | June-2019 |                                                                     |
+---------+-----------+---------------------------------------------------------------------+
|         |           | - Fix training in Docker                                            |
|         |           |                                                                     |
| 0.1.1   | Feb-2019  |                                                                     |
+---------+-----------+---------------------------------------------------------------------+
|         |           | - First stable release                                              |
|         |           | - Training/Inference examples                                       |
| 0.1.0   | Jan-2019  |                                                                     |
+---------+-----------+---------------------------------------------------------------------+

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
