from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
import os
import pickle
import yaml

from IPython import embed
import matplotlib.pyplot as plt


def state_list_key(media_by_id, state):
    for mv in state.media:
        if mv in media_by_id:
            break
    else:
        raise ValueError(f"None of {state.media} found in map")

    return (media_by_id[mv].name, state.frame)


def gt_state_list_key(media_by_id, state):
    for mv in state.media:
        if mv in media_by_id:
            break
    else:
        raise ValueError(f"None of {state.media} found in map")

    return (media_by_id[mv].name, state.attributes["Start Frame"], state.attributes["End Frame"])


def filename_to_timestamp(filename):
    return datetime.strptime(
        "_".join(os.path.splitext(file_name)[0].split("_")[1:]), "%Y%m%d_%H%M%S"
    )


def state_to_timestamp(media_by_id, state):
    filename, start_frame = state_list_key(media_by_id, state)
    return filename_to_timestamp(filename) + timedelta(seconds=start_frame / 15)


def states_to_data(media_by_id, states, activities):
    data = {"timestamps": [], "scores": defaultdict(list)}

    for state in states:
        data["timestamps"].append(state_to_timestamp(media_by_id, state))
        for activity in activities:
            if activity in state.attributes:
                data["scores"][activity].append(state.attributes[activity])

    return data


def gt_state_to_timestamps(media_by_id, state):
    filename, start_frame, end_frame = gt_state_list_key(media_by_id, state)
    file_start_time = filename_to_timestamp(filename)
    start_time = file_start_time + timedelta(seconds=start_frame / 15)
    end_time = file_start_time + timedelta(seconds=end_frame / 15)
    return start_time, end_time


def append_gt_scores(scores, gt_activity, activities):
    for activity in activities:
        scores[activity].append(1.0 if activity == gt_activity else 0.0)


def gt_states_to_data(media_by_id, gt_states, activities):
    data = {"timestamps": [], "scores": {activity: [] for activity in activities}}

    for state in gt_states:
        activity = state.attributes["Activity Type"]
        if activity not in activities:
            continue

        start_time, end_time = gt_state_to_timestamps(media_by_id, state)

        # Create background activity before start of actual activity
        data["timestamps"].append(start_time - timedelta(seconds=1))
        append_gt_scores(data["scores"], "Background", activities)

        # Create start of activity
        data["timestamps"].append(start_time)
        append_gt_scores(data["scores"], activity, activities)

        # Create end of activity
        data["timestamps"].append(end_time)
        append_gt_scores(data["scores"], activity, activities)

        # Create background activity after end of actual activity
        data["timestamps"].append(end_time + timedelta(seconds=1))
        append_gt_scores(data["scores"], "Background", activities)

    return data


def collect_data(
    api,
    *,
    project_id,
    state_type,
    gt_state_type,
    gt_versions,
    media_type,
    versions,
    trips,
    activities,
    destination_folder,
    data_filename,
):
    print("Getting sections")
    section_list = [sec for sec in api.get_section_list(project_id) if sec.name in trips]
    print("done!")
    print("Getting trip media")
    trip_media = {
        sec.name: [
            mv
            for mv in api.get_media_list(project_id, type=media_type, section=sec.id)
            if mv.media_files is not None and len(mv.media_files.ids) == 3
        ]
        for sec in section_list
    }
    print("done!")
    trip_ids = {sec_name: {mv.id: mv for mv in mv_lst} for sec_name, mv_lst in trip_media.items()}

    for sec_ids in trip_ids.values():
        sec_ids_update = {}
        for mv in sec_ids.values():
            sec_ids_update.update((m_id, mv) for m_id in mv.media_files.ids)
        sec_ids.update(sec_ids_update)

    print("Getting states for target versions")
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
    print("done!")

    print("Getting states for baseline version")
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
    print("done!")

    print("Saving data")
    del section_list, trip_media, trip_ids
    with open(os.path.join(destination_folder, data_filename), "wb") as fp:
        pickle.dump(trip_states, fp)
    print("done!")


def plot_data(destination_folder, activities, data_filename):
    with open(os.path.join(destination_folder, data_filename), "rb") as fp:
        trip_states = pickle.load(fp)

    for trip_name, trip_data in trip_states.items():
        n_versions = len(trip_data)
        if n_versions == 0:
            print(f"No trip data found for {trip_name}, skipping")
            continue
        fig, axs = plt.subplots(n_versions, 1, sharex=True, tight_layout=True)
        if n_versions == 1:
            axs = [axs]
        for (version_name, version_states), ax in zip(trip_data.items(), axs):
            is_baseline = version_name == "Baseline"
            zeroes = [0] * len(version_states["timestamps"]) if is_baseline else None
            for activity in activities:
                if is_baseline:
                    ax.fill_between(
                        version_states["timestamps"],
                        zeroes,
                        version_states["scores"][activity],
                        label=activity,
                    )
                else:
                    ax.plot(
                        version_states["timestamps"],
                        version_states["scores"][activity],
                        label=activity,
                    )

            fw = "bold" if trip_name.startswith(version_name) else "normal"
            sp_title = version_name if is_baseline else f"{version_name} model"
            ax.set_title(sp_title, fontweight=fw)

        fig.text(0.025, 0.95, trip_name)
        axs[-1].legend()
        plt.savefig(os.path.join(destination_folder, f"{trip_name}.png"), dpi=256)
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_filename", type=str)
    args = parser.parse_args()

    with open(args.config_filename, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    if config["collect_data"]:
        import tator

        api = tator.get_api(**config["tator"])
        collect_data(api, **config["data_config"])

    plot_data(
        config["data_config"]["destination_folder"],
        config["data_config"]["activities"],
        config["data_config"]["data_filename"],
    )
