Classification
==============

`Tator <https://github.com/cvisionai/tator>`_  is a web-based media management
and curation project. Part of the media management is executing algorithms or
*workflows* on a set of media. OpenEM is able to be run within the confines
of a Tator workflow. Currently Retinanet-based Detection is supported for
inference within a workflow.

Using the Reference Classification Workflow
*******************************************

The reference workflow can be used by modifying the
`scripts/tator/classification_workflow.yaml` to match those of the given
project.

Generating a data image
^^^^^^^^^^^^^^^^^^^^^^^

The reference workflow at run-time pulls a docker image containing network
coefficients and weights. To generate a weights image, one can use the
`scripts/make_classification_image.py` in a manner similar to below:

.. code-block:: shell
   :linenos:

   python3 make_classification_image.py [-h] [--publish] [--image-tag IMAGE_TAG] models [models ...]

   positional arguments:
      models                One or more models to incorporate into image.

   optional arguments:
      -h, --help            show this help message and exit
      --publish             If supplied pushes to image repo
      --image-tag IMAGE_TAG
                            Name of image to build/publish

Using the reference workflow definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A reference workflow yaml is in the repository which can be modified to
indicate project-specific requirements. Arguments in the ``tator`` section refer
to tator-level semantics such as the ``track_type_id`` to acquire thumbnails from
and the attribute name to use, to output predictions ``label_attribute``.

Options in the ``ensemble_config`` section map to the arguments and defaults
used to initialize ``openem2.Classifier.thumbnail_classifier.EnsembleClassifier``

Options to ``track_params`` section map to the arguments and defaults to the
``process_track_results`` function of the instantiated ``EnsembleClassifier``.

.. literalinclude:: ../scripts/tator/classification_workflow.yaml
   :linenos:
   :language: yaml
   :emphasize-lines: 26-37
                     
Project setup
^^^^^^^^^^^^^

A project for using this workflow has a video type represented by a ``<media_type_id>``. The project also has a localization box type represented by
``<box_type_id>``. The project has a ``<track_type_id>`` that associates multiple localizations as the same physical object.

The
``<media_type_id>>`` has the following required attributes:

.. glossary::

   Track Classification Processed 
     A string attribute type that is set to the date time when the object
     detector finishes processing the media file.

The ``<track_type_id>`` requires the following attributes:

.. glossary::

   <label_attribute>
     A string representing the name for an object class. If 'Label' is not
     an appropriate name for class, this can be customized via the
     ``label_attribute`` key in the strategy definition.

   Entropy
     This float attribute represents the uncertainty of the classification
     algorithm in its determination.
