#!/usr/bin/env python3

import argparse
import yaml
import logging
from statistics import median

import tator

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tator.get_parser(parser)
    parser.add_argument("--tracklet-type-id", type=int, required=True)
    parser.add_argument("--version-id", type=int)
    parser.add_argument("--attribute-name", type=str)
    parser.add_argument("--strategy-config", type=str)
    parser.add_argument('media_ids', type=int, nargs='+')
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
                        "args": {}}

    if args.strategy_config:
        strategy = {**default_strategy}
        with open(args.strategy_config, "r") as strategy_file:
            strategy.update(yaml.load(strategy_file))
    else:
        strategy = default_strategy

    dimension = strategy["dimension"]
    method = strategy["method"]
    transform = strategy["transform"]
    medias = api.get_media_list_by_id(project, {'ids': args.media_ids})
    medias = {media.id:media for media in medias}
    tracks = api.get_state_list_by_id(project, {'media_ids': args.media_ids},
                                      type=tracklet_type.id, version=[version_id])
    for track in tracks:
        locs = api.get_localization_list_by_id(project, {'ids': track.localizations})
        media = medias[track.media[0]]
        if dimension == "both":
            sizes = [(loc.width * media.width + loc.height * media.height) / 2 for loc in locs]
        elif dimension == "width":
            sizes = [loc.width * media.width for loc in locs]
        elif dimension == "height":
            sizes = [loc.height * media.height for loc in locs]
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
        if transform == "scallops":
            size = ((0.1/120) * (size - 40) + 0.8) * size
        elif transform == "none":
            pass
        else:
            raise ValueError(f"Invalid transform \'{transform}\', must be one of "
                              "\'scallops\' or \'none\'!")
        print(f"Updating track {track.id} with {attribute_name} = {size}...")
        response = api.update_state(track.id, {'attributes': {attribute_name: size}})
        assert(isinstance(response, tator.models.MessageResponse))
        print(response.message)

