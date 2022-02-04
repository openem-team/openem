from argparse import ArgumentParser
from collections import defaultdict
from configparser import ConfigParser
from datetime import datetime, timedelta
import json
import logging
import os
from pprint import pformat
import random

import boto3
from IPython import embed
import pandas as pd

import tator

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)

ACTIVITY_LIST = ["Fish Presence", "Fish Activity", "Net Activity", "Offloading", "Background"]


def _make_activity_dict():
    return {activity: [] for activity in ACTIVITY_LIST}


def get_state_list(api, project_id, state_type, min_cameras=3, bad_media=None):
    bad_media = bad_media or []
    return [
        state
        for state in api.get_state_list(project_id, type=state_type)
        if len(state.media) >= min_cameras and not any(m in bad_media for m in state.media)
    ]


def _download_feature(client, target_folder, s3_info):
    bucket = s3_info["bucket"]
    key = s3_info["key"]
    target_filename = os.path.join(target_folder, key)
    if not os.path.isfile(target_filename):
        if client.list_objects_v2(Bucket=bucket, Prefix=key)["KeyCount"] == 1:
            logger.info(f"Downloading {key}")
            client.download_file(bucket, key, target_filename)
        else:
            logger.warning(f"Could not find {key} in bucket {bucket}")

    return key


def check_raw_data(api, project_id, state_type, min_cameras, bad_media):
    """ Logs stats about raw data """
    state_list = get_state_list(api, project_id, state_type, min_cameras, bad_media)
    all_section_list = [sec for sec in api.get_section_list(project_id) if sec.tator_user_sections]

    multi_section_list = [sec for sec in all_section_list if sec.name.endswith("_Trip")]
    section_dict = {sec.tator_user_sections: sec for sec in all_section_list}

    for sec in multi_section_list:
        multis = api.get_media_list(project=project_id, section=sec.id)
        seconds = 0
        for media in multis:
            if media.media_files.ids is None:
                continue
            singles = api.get_media_list(project=project_id, media_id=media.media_files.ids)
            seconds += min(s.num_frames / s.fps for s in singles)

        # TODO
        # 2. Do something with `seconds`
        # 3. ???
        # 4. Profit!


def check_and_download_features(api, client, project_id, state_type, target_folder, bad_media):
    state_list = get_state_list(api, project_id, state_type, min_cameras=3, bad_media=bad_media)

    media_id_list = list(set(m_id for state in state_list for m_id in state.media))
    missing_features = []
    feature_files = {}
    chunk = 250
    for offset in range(0, len(media_id_list), chunk):
        for media in api.get_media_list(
            project_id, media_id=media_id_list[offset : offset + chunk]
        ):
            json_str = media.attributes["feature_s3"]
            try:
                feature_files[media.id] = _download_feature(
                    client, target_folder, json.loads(json_str)
                )
            except json.JSONDecodeError:
                logger.warning(f"Could not get features for {media.id}, invalid json '{json_str}'")
                missing_features.append(media.id)
            except BaseException as e:
                logger.error(e)

    logger.info(f"{len(missing_features)} media missing features")
    if len(missing_features):
        logger.info(missing_features)
    return feature_files


def gen_sample_starts(api, project_id, state_type, bad_media):
    state_list = get_state_list(api, project_id, state_type, min_cameras=3, bad_media=bad_media)

    section_by_media_id = {}
    section_list = [
        section for section in api.get_section_list(project_id) if section.tator_user_sections
    ]
    media_id_list = list(set(media for state in state_list for media in state.media))
    media_frames = {}
    chunk = 250
    for offset in range(0, len(media_id_list), chunk):
        for media in api.get_media_list(
            project_id, media_id=media_id_list[offset : offset + chunk]
        ):
            section = next(
                sec
                for sec in section_list
                if sec.tator_user_sections == media.attributes["tator_user_sections"]
            )
            vessel, trip_date = section.name.split("-")
            section_by_media_id[media.id] = {
                "id": section.id,
                "tator_user_sections": section.tator_user_sections,
                "trip_date": trip_date,
                "vessel": vessel,
            }
            media_frames[media.id] = media.num_frames

    sample_starts = {}
    fps = 15
    sample_time = 1  # seconds
    n_sample_frames = fps * sample_time
    for state in state_list:
        section = section_by_media_id[state.media[0]]
        vessel = section["vessel"]
        trip_date = section["trip_date"]

        if vessel not in sample_starts:
            sample_starts[vessel] = {}
        vessel_samples = sample_starts[vessel]
        if trip_date not in vessel_samples:
            vessel_samples[trip_date] = _make_activity_dict()
        trip_samples = vessel_samples[trip_date]
        state_start = state.attributes["Start Frame"]
        state_end = state.attributes["End Frame"]
        max_frames = min(media_frames[media_id] for media_id in state.media)
        last_frame = min(max_frames, state_end - fps * 60)
        state_samples = [
            {
                "media_ids": state.media,
                "start_frame": sample_start,
            }
            for sample_start in range(state_start, last_frame)
        ]
        trip_samples[state.attributes["Activity Type"]].extend(state_samples)

    return sample_starts


def gen_tv_splits(sample_starts):
    val_trips = {}

    for vessel, trip_dict in sample_starts.items():
        logger.info(f"Calculating validation set for {vessel}...")
        n_trips = len(trip_dict)
        n_val_trips = int(n_trips * 0.15)

        val_trips[vessel] = random.sample(trip_dict.keys(), n_val_trips)

    log_tv_stats(val_trips, sample_starts)
    return val_trips


def log_tv_stats(val_trips, sample_starts):
    fps = 15
    logger.info(f"|{'Vessel': >6}|{'Activity': >16}|{'Train': >5}|{'Val': >4}|{'Total': >5}|")
    for vessel in val_trips.keys():
        n_val_samples = defaultdict(int)
        n_total_samples = defaultdict(int)
        for trip_date, trip_samples in sample_starts[vessel].items():
            for k, v in trip_samples.items():
                state_set = set(tuple(ele["media_ids"]) for ele in v)
                new_samples = len(v) + len(state_set) * 60
                if trip_date in val_trips[vessel]:
                    n_val_samples[k] += new_samples
                n_total_samples[k] += new_samples

        for k in n_val_samples.keys():
            val = round(n_val_samples[k] / 60 / fps)
            total = round(n_total_samples[k] / 60 / fps)
            activity_name = f"{k: >16}"
            logger.info(
                f"|{vessel: >6}|{activity_name}|{total - val: >5}|{val: >4}|{total: >5}|{val/total}"
            )


def gen_tv_samples(val_trips, sample_starts, feature_map, samples_by_vessel):
    # Split sample starts into train/val sets
    all_tv_starts = {}
    for vessel in val_trips.keys():
        val_dates = val_trips[vessel]
        if vessel not in all_tv_starts:
            all_tv_starts[vessel] = {"train": _make_activity_dict(), "val": _make_activity_dict()}
        vessel_tv = all_tv_starts[vessel]
        for trip_date, trip_dict in sample_starts[vessel].items():
            if trip_date in val_dates:
                dest_dict = vessel_tv["val"]
            else:
                dest_dict = vessel_tv["train"]
            for activity_name, activity_list in trip_dict.items():
                dest_dict[activity_name].extend(activity_list)

    # Sample from train/val sets
    sampled_tv_starts = {}
    for vessel, vessel_dict in all_tv_starts.items():
        if vessel not in sampled_tv_starts:
            sampled_tv_starts[vessel] = {
                "train": _make_activity_dict(),
                "val": _make_activity_dict(),
            }
        for tv_name, tv_dict in vessel_dict.items():
            n_samples = samples_by_vessel[vessel][tv_name]
            sample_sets = sampled_tv_starts[vessel][tv_name]
            for activity_name, activity_list in tv_dict.items():
                try:
                    sample_sets[activity_name].extend(
                        random.sample(activity_list, min(len(activity_list), n_samples))
                    )
                except:
                    embed()
                    raise

    return sampled_tv_starts


def _extract_single_sample(feature_map, download_folder, media_ids, start_frame):
    id_to_feature = {
        media_id: pd.read_hdf(os.path.join(download_folder, feature_map[str(media_id)]))
        for media_id in media_ids
    }
    sample_features_df = pd.concat(
        [
            pd.concat([id_to_feature[media_id].iloc[frame] for media_id in media_ids])
            for frame in range(start_frame, start_frame + 60 * 15, 15)
        ],
        axis=1,
    ).transpose()

    sample_features_df.columns = list(range(len(sample_features_df.columns)))
    return sample_features_df


def extract_tv_samples(tv_samples, feature_map, download_folder, destination_folder):
    for vessel, vessel_dict in tv_samples.items():
        for tv_name, tv_dict in vessel_dict.items():
            for activity_name, activity_list in tv_dict.items():
                activity_str = activity_name.replace(" ", "_")
                for idx, sample in enumerate(activity_list):
                    filename = f"{tv_name}__{vessel}__{activity_str}__{idx:04d}.hdf"
                    sample_dest = os.path.join(destination_folder, filename)
                    if os.path.isfile(sample_dest):
                        continue
                    try:
                        sample_features_df = _extract_single_sample(
                            feature_map, download_folder, sample["media_ids"], sample["start_frame"]
                        )
                    except:
                        logger.warning(f"Could not extract features for {filename}", exc_info=True)
                    else:
                        sample_features_df.to_hdf(sample_dest, f"idx{idx}")


def check_tv_files(destination_folder, samples_by_vessel):
    # stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    stats = {}

    for filename in os.listdir(destination_folder):
        tv, vessel, activity, idx = os.path.splitext(filename)[0].split("__")
        if vessel not in stats:
            stats[vessel] = {}
        if tv not in stats[vessel]:
            stats[vessel][tv] = {}
        if activity not in stats[vessel][tv]:
            stats[vessel][tv][activity] = 0

        # stats[vessel][tv][activity].add(int(idx))
        stats[vessel][tv][activity] += 1

    logger.info(f"STATS:\n{pformat(stats)}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract samples from a Tator project")
    parser.add_argument("config_file", help="Path to config file.")
    args = parser.parse_args()

    # Read the config file.
    config = ConfigParser(os.environ)
    config.read(args.config_file)
    project_id = config["Media"]["ProjectId"]
    state_type = config["Media"]["StateType"]
    bad_media_str = config["Media"]["BadMedia"]
    bad_media = [int(ele) for ele in bad_media_str.split(",")] if bad_media_str else []
    feature_map_file = config["FeatureExtraction"]["FeatureMapFile"]
    download_folder = config["FeatureExtraction"]["DownloadFolder"]
    sample_starts_file = config["CollectSamples"]["SampleStartsFile"]
    val_trips_file = config["GenValTrips"]["ValTripsFile"]
    tv_samples_file = config["GenSamples"]["TvSamplesFile"]
    destination_folder = config["ExtractSamples"]["DestinationFolder"]

    # Default values
    feature_map = None
    sample_starts = None
    val_trips = None
    tv_samples = None

    api = tator.get_api(
        token=config["Credentials"]["TatorToken"], host=config["Credentials"]["TatorHost"]
    )
    client = boto3.client(
        "s3",
        endpoint_url=config["Credentials"]["AwsEndpointUrl"],
        aws_access_key_id=config["Credentials"]["AwsAccessKey"],
        aws_secret_access_key=config["Credentials"]["AwsSecretKey"],
    )

    if config.getboolean("CheckRawData", "Run"):
        check_raw_data(
            api, project_id, state_type, config.getint("CheckRawData", "MinCameras"), bad_media
        )

    if config.getboolean("FeatureExtraction", "Run"):
        feature_map = check_and_download_features(
            api, client, project_id, state_type, download_folder, bad_media
        )
        with open(feature_map_file, "w") as fp:
            json.dump(feature_map, fp, indent=2)

    if config.getboolean("CollectSamples", "Run"):
        sample_starts = gen_sample_starts(api, project_id, state_type, bad_media)
        with open(sample_starts_file, "w") as fp:
            json.dump(sample_starts, fp, indent=2)

    if config.getboolean("GenValTrips", "Run"):
        if sample_starts is None:
            with open(sample_starts_file, "r") as fp:
                sample_starts = json.load(fp)

        val_trips = gen_tv_splits(sample_starts)
        with open(val_trips_file, "w") as fp:
            json.dump(val_trips, fp, indent=2)

    if config.getboolean("GenValTrips", "TestTv"):
        if sample_starts is None:
            with open(sample_starts_file, "r") as fp:
                sample_starts = json.load(fp)
        if val_trips is None:
            with open(val_trips_file, "r") as fp:
                val_trips = json.load(fp)

        log_tv_stats(val_trips, sample_starts)

    samples_by_vessel = json.loads(config["Media"]["SamplesByVessel"])

    if config.getboolean("GenSamples", "Run"):
        if sample_starts is None:
            with open(sample_starts_file, "r") as fp:
                sample_starts = json.load(fp)
        if val_trips is None:
            with open(val_trips_file, "r") as fp:
                val_trips = json.load(fp)
        if feature_map is None:
            with open(feature_map_file, "r") as fp:
                feature_map = json.load(fp)

        tv_samples = gen_tv_samples(val_trips, sample_starts, feature_map, samples_by_vessel)
        with open(tv_samples_file, "w") as fp:
            json.dump(tv_samples, fp, indent=2)

    if config.getboolean("ExtractSamples", "Run"):
        if tv_samples is None:
            with open(tv_samples_file, "r") as fp:
                tv_samples = json.load(fp)
        if feature_map is None:
            with open(feature_map_file, "r") as fp:
                feature_map = json.load(fp)

        extract_tv_samples(tv_samples, feature_map, download_folder, destination_folder)

    if config.getboolean("ExtractSamples", "CheckTv"):
        check_tv_files(destination_folder, samples_by_vessel)
