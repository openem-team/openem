from argparse import ArgumentParser
from datetime import datetime
import json
from typing import Dict
import yaml

import tator


def check_inferred_states(
    api, project_id: int, media_type: int, trips: list, versions: Dict[str, int]
):
    section_list = api.get_section_list(project_id)
    all_stats = {}
    t_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    summary_lines = []
    for section in section_list:
        section_name = section.name
        if section_name not in trips:
            continue

        for version_name, version_ids in versions.items():
            if section_name.startswith(version_name):
                version = version_ids
                break
        else:
            msg = f"No matching version found for trip '{section_name}'"
            print(msg)
            summary_lines.append(msg)
            continue

        msg = f"Stats for trip {section_name}:"
        summary_lines.append(msg)
        print(msg)

        stats = {
            "media ok": [],
            "too many states": [],
            "too few states": [],
            "too few videos": [],
            "missing feature_s3": [],
        }
        for media in api.get_media_list(project_id, type=media_type, section=section.id):
            video_ids = media.media_files.ids
            state_count = api.get_state_count(project_id, media_id=[media.id], version=version)
            if video_ids is None or len(video_ids) < 3:
                stats["too few videos"].append(media.id)
            elif any(not api.get_media(mid).attributes["feature_s3"] for mid in video_ids):
                stats["missing feature_s3"].append(media.id)
            elif state_count < 14:
                stats["too few states"].append(media.id)
            elif state_count > 15:
                stats["too many states"].append(media.id)
            else:
                stats["media ok"].append(media.id)

        for status, ids in stats.items():
            msg = f"{status}: {len(ids)}"
            summary_lines.append(msg)
            print(msg)

        all_stats[section_name] = stats

    with open(f"{t_str}_summary_stats.txt", "w") as summary_fp:
        summary_fp.write("\n".join(summary_lines))

    with open(f"{t_str}_full_stats.json", "w") as fp:
        json.dump(all_stats, fp, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-filename", type=str, default="check_inferred_states_config.yaml")
    args = parser.parse_args()

    with open(args.config_filename, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    check_inferred_states(
        tator.get_api(**config["tator"]),
        config["data_config"]["project_id"],
        config["data_config"]["media_type"],
        config["data_config"]["trips"],
        config["data_config"]["versions"],
    )
