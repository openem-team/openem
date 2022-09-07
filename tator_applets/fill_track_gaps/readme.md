# Automated Fill Track Gaps
This Tator applet enables annotators that ability to right click on a track detection and launch
a workflow that will fill the track's coasted frames (i.e. gaps) using the selected OpenEM-centric
algorithm.

To utilize this applet in a particular Tator project, follow the installation instructions below.

# Installation

## Applet
Register this [applet](https://github.com/openem-team/openem/blob/master/tator_applets/fill_track_gaps/fill_track_gaps_applet.html)in Tator using the project settings page or via tator-py.  

The applet must be registered with the following categories:
- annotator-menu
- track-applet
- openem-csrt
- openem-linear

The openem categories above can be omitted based on the project setup. These categories enable the Linear and Visual based options in the applet.

The name of the applet and its description can be modified to the user's preference. Note: The name of the applet is what's displayed in the right-click menu. Since this applet has the `track-applet` category, the applet will only appear when the annotator right clicks on a track on the video canvas.

## Workflow
Register this [workflow](https://github.com/openem-team/openem/blob/master/tator_applets/fill_track_gaps/openem_fill_track_gaps.yaml) in Tator using tator-py. Utilize the [register_algorithm.py](https://github.com/cvisionai/tator-py/blob/9cc5e8240e3c8761a90080b977dec25bccf68b39/examples/register_algorithm.py) Python utility to register the algorithm.

The algorithm should be registered with the following categories:
- annotator-view
- hidden

The name of the algorithm must be `openem-fill-track-gaps`. The applet looks for a registered algorithm with this name.

Example usage:
python3 ./register_algorithm.py --host <tator_host> --token <tator_api_token> --project <project-id> --manifest openem_fill_track_gaps.yaml --files_per_job 1 --description "OpenEM algorithm that is used by the fill track applet" --categories annotator-view hidden --algorithm_name openem-fill-track-gaps

## Tator Version
This applet requires using (this Tator commit)[https://github.com/cvisionai/tator/tree/3e3c13684237657a3e27005d9443e3ae04bc76d8] at minimum. This commit contains the framework to utilize track specific applets.
