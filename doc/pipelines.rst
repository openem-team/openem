OpenEM and Tator Pipelines
==========================

`Tator <https://github.com/cvisionai/tator>`_  is a web-based media management
and curation project. Part of the media mangement is executing algorithms or
*workflows* on a set of media. OpenEM is able to be run within the confines
of a Tator workflow. Currently Retinanet-based Detection is supported for
inference within a workflow.

Using the Reference Detection Workflow
**************************************

The reference workflow can be used by modifying the
`scripts/tator/detection_workflow.yaml` to match those of the given project.

Generating a data image
^^^^^^^^^^^^^^^^^^^^^^^

The reference workflow at run-time pulls a docker image containing network
coefficients and weights. To generate a weights image, one can use the
`scripts/make_pipeline_image.py` in a manner similar to below:

.. code-block:: shell
   :linenos:

   python3 make_pipeline_image.py --graph-pb <trained.pb> --train-ini <path_to_train.ini> --publish <docker_hub_user>/<image_name>

Note the values of <docker_hub_user> and <image_name> for use in the next
section.

The referenced train.ini can be a subset of full `train.ini`;
a minimimal configuration such as the following is acceptable for the
requirements of `uploadToTator.py`:

.. code-block:: ini
   :linenos:

   [Data]
   # Names of species, separated by commas.
   Species=Fish


Using the reference workflow definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A reference workflow yaml is in the repository which can be modified to
indicate project-specific requirements. `img_max_side`, `img_min_side`,
`batch_size`, and `keep_threshold` map to the arguments in `infer.py` directly.

This workflow is for executing retinanet-based detections on a video dataset
using tensor-rt enabled hardware.

Nominally the only parameters required to change are the `TATOR_PIPELINE_ARGS` for each stage of the workflow.

.. literalinclude:: ../scripts/tator/detection_workflow.yaml
   :linenos:
   :language: yaml
   :emphasize-lines: 48, 74, 99
                     
Detailed Mechanics
******************

This section walks through the mechanics of the reference workflow so that
users could build more ellaborate workflows on OpenEM technology.

A Tator Workflow is specifed no differently than a regular `Argo <https://argoproj.github.io/argo/>`_ workflow, other than there is an expectation the Tator REST
API is used to access media files and supply results to a project.

A canonoical Tator workflow has three parts: setup, execution, and teardown.
More advanced workflows can replace the execution stage with multiple stages
using the directed acyclic graph capabilities of argo.

Project setup
^^^^^^^^^^^^^

A project for using this workflow has a media type (either a video type or
an image type) represented by a ``<media_type_id>``. The project also has a
localization box type represented by ``<box_type_id>``. The
``<media_type_id>>`` has the following required attributes:

.. glossary::

   Object Detector Processed
     A string attribute type that is set to the date time when the object
     detector finishes processing the media file.

The ``<box_type_id>`` requires the the following attributes:

.. glossary::

   Species
     A string representing the name for an object class. If 'Species' is not
     an appropriate name for class, this can be customized via the
     ``species_attr_name`` key in the pipeline argument object to the
     teardown stage. It defaults to 'Species' if not specified.

   Confidence
     A float attribute representing the score of the detection. If 'Confidence'
     is not a desired name, it can be customized via the
     ``confidence_attr_name`` key in the pipeline argument object to the
     teardown stage. It defaults to 'Confidence' if not specified.

Acquiring media
^^^^^^^^^^^^^^^

The example `setup.py` provides a canonical way to download media for a given
workflow.

.. literalinclude:: ../scripts/tator/setup.py
   :linenos:
   :language: python

Executing Work
^^^^^^^^^^^^^^^

The heart of the reference workflow is `infer.py` from the openem_lite docker
image. However, it is useful to have a layer of scripting above that CLI
utility to translate workflow definitions to the underlying utility.

.. literalinclude:: ../scripts/tator/docker_entry.py
   :linenos:
   :language: python

Submitting results
^^^^^^^^^^^^^^^^^^

`infer.py` generates a csv with inference results, so another utility must
interpret these results and submit to the underlying Tator web service. A script
called `uploadToTator.py` is located in scripts, but similar to `infer.py`;
inserting a layer between the raw script can be helpful to mananage
the environment.

.. literalinclude:: ../scripts/tator/teardown.py
   :linenos:
   :language: python
