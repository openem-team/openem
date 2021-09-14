#!/usr/bin/env python3

import argparse
import yaml
import logging
from statistics import median

import tator

logger = logging.getLogger(__name__)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tator.get_parser(parser)
    parser.add_argument("--tracklet-type-id", type=int, required=True)
    parser.add_argument("--version-id", type=int)
    parser.add_argument("--attribute-name", type=str)
    parser.add_argument("--strategy-config", type=str)
    parser.add_argument("--dry-run", action='store_true')
    parser.add_argument('media_files', type=str, nargs='*')
    args = parser.parse_args()

    # Weight methods
    methods = ['mean', 'median']

    api = tator.get_api(args.host, args.token)
    tracklet_type = api.get_state_type(args.tracklet_type_id)
    project = tracklet_type.project
    version_id = args.version_id
    attribute_name = args.attribute_name

    default_strategy = {"method": "median",
                        "dimension": "both",
                        "scale-factor": 1.0, 
                        "args": {},

    if args.strategy_config:
        strategy = {**default_strategy}
        with open(args.strategy_config, "r") as strategy_file:
            strategy.update(yaml.load(strategy_file))
    else:
        strategy = default_strategy

    tracks = api.get_state_list(project, type=tracklet_type, version=version_id)
    for track in tracks:
        locs = api.get_localization_list_by_id(project, ids=track.localizations)
        dimension = strategy["dimension"]
        if dimension == "both":
            sizes = [(loc.width + loc.height) / 2 for loc in locs]
        elif dimension == "width":
            sizes = [loc.width for loc in locs]
        elif dimension == "height":
            sizes = [loc.height for loc in locs]
        else:
            raise ValueError(f"Invalid dimension \'{dimension}\', must be one of "
                              "\'width\', \'height\', or \'both\'!")
        if method == "median":
            size = median(sizes)
        elif method == "mean":
            size = sum(sizes) / len(sizes)
        else:
            raise ValueError(f"Invalid method \'{method}\', must be one of "
                              "\'median\' or \'mean\'!")
        size *= strategy['scale-factor']
        logger.info(f"Updating track {track.id} with {attribute_name} = {size}...")
        response = api.update_state(track.id, {attribute_name: size})
        assert(isinstance(response, tator.models.MessageResponse))

