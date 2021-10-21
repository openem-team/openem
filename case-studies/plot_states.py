from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
import logging
import os
from pathlib import Path
import pickle
from typing import List, Dict
import yaml

from IPython import embed
import matplotlib.pyplot as plt

import tator


logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


def filename_to_timestamp(filename):
    """
    Turns a filename into a datetime object. Assumes `filename` is of the format
    `<extra>_<%Y%m%d>_<%H%M%S>.<extension>`
    """
    return datetime.strptime(
        "_".join(os.path.splitext(filename)[0].split("_")[1:]), "%Y%m%d_%H%M%S"
    )


def state_list_key(media_by_id, state):
    """
    Gets the start time of the media file in which the inferred state occurs, followed by the start
    time of the state itself. Can be used as a key function for sorting.
    """
    for mv in state.media:
        if mv in media_by_id:
            break
    else:
        raise ValueError(f"None of {state.media} found in map")

    file_start_time = filename_to_timestamp(media_by_id[mv].name)
    start_time = file_start_time + timedelta(seconds=state.frame / 15)
    return file_start_time, start_time


def gt_state_list_key(media_by_id, state):
    """
    Gets the start time of the media file in which the hand-annotated state occurs, followed by the
    start time of the state itself. Can be used as a key function for sorting.
    """
    for mv in state.media:
        if mv in media_by_id:
            break
    else:
        raise ValueError(f"None of {state.media} found in map")

    file_start_time = filename_to_timestamp(media_by_id[mv].name)
    start_time = file_start_time + timedelta(seconds=state.attributes["Start Frame"] / 15)
    end_time = file_start_time + timedelta(seconds=state.attributes["End Frame"] / 15)
    return file_start_time, start_time, end_time


def get_frame_count(media_by_id, state):
    """ Gets the frame count of the shortest video containing the given state """
    frame_counts = [
        media_by_id[mv].num_frames
        for mv in state.media
        if mv in media_by_id and media_by_id[mv].num_frames
    ]

    if frame_counts:
        return min(frame_counts)
    return 0


def states_to_data(media_by_id, states, activities):
    """
    Converts the list of inferred states into a list of timestamps (x values for plotting) and lists
    of scores separated by activity.
    """
    data = {"timestamps": [], "scores": {activity: [] for activity in activities}}

    for state in states:
        data["timestamps"].append(state_list_key(media_by_id, state)[1])
        for activity in activities:
            data["scores"][activity].append(state.attributes.get(activity, -1))

    return data


def append_gt_scores(scores, gt_activity):
    """ Appends scores for each activity; 1 for the given activity, 0 otherwise """
    for activity_name, activity in scores.items():
        activity.append(1.0 if activity_name == gt_activity else 0.0)


def gt_states_to_data(media_by_id, gt_states, activities):
    """
    Converts the list of hand annotated states into a list of timestamps (x values for plotting) and
    lists of scores separated by activity.
    """
    data = {
        "timestamps": [],
        "file_starts": {"timestamps": [], "ids": []},
        "scores": {activity: [] for activity in activities},
    }

    if not gt_states:
        return data

    # Start the video with a background event
    trip_start_time = gt_state_list_key(media_by_id, gt_states[0])[0]
    data["timestamps"].append(trip_start_time)
    append_gt_scores(data["scores"], "Background")

    # Since each state is annotated with a start and an end frame, we need to generate a score for
    # each of them. Additionally, we need to book-end these scores with a Background event, to show
    # that there is nothing happening between events.
    for state in gt_states:
        activity = state.attributes["Activity Type"]
        if activity not in activities:
            continue

        # Get start time of the file containing this state
        file_start_time, start_time, end_time = gt_state_list_key(media_by_id, state)
        mv_id = media_by_id[state.media[0]].id
        if mv_id not in data["file_starts"]["ids"]:
            data["file_starts"]["timestamps"].append(file_start_time)
            data["file_starts"]["ids"].append(mv_id)

        # Create background activity before start of actual activity
        data["timestamps"].append(start_time - timedelta(seconds=1))
        append_gt_scores(data["scores"], "Background")

        # Create start of activity
        data["timestamps"].append(start_time)
        append_gt_scores(data["scores"], activity)

        # Create end of activity
        data["timestamps"].append(end_time)
        append_gt_scores(data["scores"], activity)

        # Create background activity after end of actual activity
        data["timestamps"].append(end_time + timedelta(seconds=1))
        append_gt_scores(data["scores"], "Background")

    # End the video with a background event
    last_video_start_time = gt_state_list_key(media_by_id, gt_states[-1])[0]
    video_length = timedelta(seconds=get_frame_count(media_by_id, gt_states[-1]) / 15)
    trip_end_time = last_video_start_time + video_length
    data["timestamps"].append(trip_end_time)
    append_gt_scores(data["scores"], "Background")

    return data


def collect_data(
    api: tator.api,
    *,
    project_id: int,
    state_type: int,
    gt_state_type: int,
    gt_versions: List[int],
    media_type: int,
    versions: Dict[str, List[int]],
    trips: List[str],
    activities: List[str],
    destination_folder: str,
    data_filename: str,
):
    """
    Collects and formats data to be plotted.

    :param api: used to retrieve data from tator
    :type api: tator.api
    :param project_id: the id of the project that the data belong to
    :type project_id: int
    :param state_type: the id of the inferred state type
    :type state_type: int
    :param gt_state_type: the id of the hand annotated state type
    :type gt_state_type: int
    :param gt_versions: the version where the hand annotated states exist
    :type gt_versions: List[int]
    :param media_type: the multivideo type id
    :type media_type: int
    :param versions: the versions where different inferred states live
    :type versions: Dict[str, List[int]]
    :param trips: the list of section names to use
    :type trips: List[str]
    :param activities: the names of the valid states
    :type activities: List[str]
    :param destination_folder: where to store the data
    :type destination_folder: str
    :param data_filename: the name of the data file to write
    :type data_filename: str
    """

    # First, make the destination folder and raise if it already exists
    Path(destination_folder).mkdir(parents=True)

    # Collect the section objects for each section name in trips
    logger.info("Getting sections")
    section_list = [sec for sec in api.get_section_list(project_id) if sec.name in trips]
    logger.info("done!")

    # Collect the multivideo objects by section
    logger.info("Getting trip media")
    trip_media = {
        sec.name: [
            mv
            for mv in api.get_media_list(project_id, type=media_type, section=sec.id)
            if mv.media_files is not None and len(mv.media_files.ids) == 3
        ]
        for sec in section_list
    }
    trip_ids = {sec_name: {mv.id: mv for mv in mv_lst} for sec_name, mv_lst in trip_media.items()}
    logger.info("done!")

    # Adds additional mappings from individual media to multivideos
    for sec_ids in trip_ids.values():
        sec_ids_update = {}
        for mv in sec_ids.values():
            sec_ids_update.update((m_id, mv) for m_id in mv.media_files.ids)
        sec_ids.update(sec_ids_update)

    # Collect the states that occur in media from target sections and inferred versions
    logger.info("Getting states for target versions")
    trip_states = {}
    for trip in trips:
        trip_states[trip] = {}
        for version in versions.keys():
            states = sorted(
                api.get_state_list(
                    project=project_id,
                    media_id=list(trip_ids[trip].keys()),
                    version=versions[version],
                    type=state_type,
                ),
                key=partial(state_list_key, trip_ids[trip]),
            )
            trip_states[trip][version] = states_to_data(trip_ids[trip], states, activities)
    logger.info("done!")

    # Collect the states that occur in media from target sections and hand annotated version
    logger.info("Getting states for baseline version")
    all_media_ids = {}
    for sec_name, mv_lst in trip_media.items():
        all_media_ids[sec_name] = []
        for mv in mv_lst:
            all_media_ids[sec_name].extend(mv.media_files.ids)

    for trip_name, media_ids in all_media_ids.items():
        if not media_ids:
            continue
        baseline_states = sorted(
            api.get_state_list(
                project=project_id, media_id=media_ids, version=gt_versions, type=gt_state_type
            ),
            key=partial(gt_state_list_key, trip_ids[trip_name]),
        )
        trip_states[trip_name]["Baseline"] = gt_states_to_data(
            trip_ids[trip_name], baseline_states, activities
        )
    logger.info("done!")

    logger.info("Saving data")
    del section_list, trip_media, trip_ids
    with open(os.path.join(destination_folder, data_filename), "wb") as fp:
        pickle.dump(trip_states, fp)
    logger.info("done!")


def plot_data(
    destination_folder: str, activities: List[str], data_filename: str, plot_file_starts: bool
):
    """

    :param destination_folder: The folder from which to read the data and write the plots
    :type destination_folder: str
    :param activities: the names of the valid states
    :type activities: List[str]
    :param data_filename: the name of the data file to write
    :type data_filename: str
    :param plot_file_starts: If true, plots the start of each multivideo containing at least one
                             state and labels it with its id.
    :type plot_file_starts: bool
    """
    with open(os.path.join(destination_folder, data_filename), "rb") as fp:
        trip_states = pickle.load(fp)

    for trip_name, trip_data in trip_states.items():
        n_versions = len(trip_data)
        if n_versions == 0:
            logger.info(f"No trip data found for {trip_name}, skipping")
            continue
        fig, axs = plt.subplots(n_versions, 1, sharex=True, tight_layout=True)
        if n_versions == 1:
            axs = [axs]
        trip_start_time = None
        trip_end_time = None
        baseline_ax = None
        for (version_name, version_states), ax in zip(trip_data.items(), axs):
            x_values = version_states["timestamps"]
            if x_values:
                first_timestamp = x_values[0]
                trip_start_time = (
                    first_timestamp
                    if trip_start_time is None
                    else min(trip_start_time, first_timestamp)
                )
                last_timestamp = x_values[-1]
                trip_end_time = (
                    last_timestamp if trip_end_time is None else min(trip_end_time, last_timestamp)
                )
            is_baseline = version_name == "Baseline"
            if is_baseline:
                baseline_ax = ax
                baseline_lines = []
                zeroes = [0] * len(x_values)
                for activity in activities:
                    l = ax.fill_between(
                        x_values,
                        zeroes,
                        version_states["scores"][activity],
                        label=activity,
                    )
                    baseline_lines.append(l)
                if plot_file_starts:
                    file_starts = version_states["file_starts"]
                    if file_starts["timestamps"]:
                        file_start_lines = [
                            ax.vlines(ts, 0, 1, linestyles="dashed", colors=["k"])
                            for ts in file_starts["timestamps"]
                        ]
                    else:
                        file_start_lines = []
                    file_start_labels = file_starts["ids"]
            else:
                for activity in activities:
                    ax.plot(
                        x_values,
                        version_states["scores"][activity],
                        label=activity,
                    )

            fw = "bold" if trip_name.startswith(version_name) else "normal"
            sp_title = version_name if is_baseline else f"{version_name} model"
            ax.set_title(sp_title, fontweight=fw)

        baseline_ax = baseline_ax or axs[-1]
        fig.text(0.025, 0.95, trip_name)
        n_ticks = 6

        if trip_start_time and trip_end_time:
            interval = (trip_end_time - trip_start_time).total_seconds() / (n_ticks - 1)
            xticks = [trip_start_time + timedelta(seconds=interval * idx) for idx in range(n_ticks)]
            baseline_ax.set_xticks(xticks)
            baseline_ax.set_xticklabels([tick.strftime("%H%M") for tick in xticks])

        legend1 = baseline_ax.legend(baseline_lines, activities, loc=3)

        if plot_file_starts:
            baseline_ax.legend(file_start_lines, file_start_labels, loc=1)
            baseline_ax.add_artist(legend1)

        plt.savefig(os.path.join(destination_folder, f"{trip_name}.png"), dpi=256)
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-filename", type=str, default="plot_states_config.yaml")
    args = parser.parse_args()

    with open(args.config_filename, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    if config["collect_data"]:
        api = tator.get_api(**config["tator"])
        collect_data(api, **config["data_config"])

    plot_data(
        config["data_config"]["destination_folder"],
        config["data_config"]["activities"],
        config["data_config"]["data_filename"],
        config["plot_file_starts"],
    )
