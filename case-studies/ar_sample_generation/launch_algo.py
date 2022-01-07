import yaml
import tator


"""
yaml file should look like this

host: https://www.tatorapp.com
token: YOUR_TOKEN_HERE
"""
config_file = "/home/ben/tatorapp.yaml"
with open(config_file, "r") as fp:
    config = yaml.load(fp, Loader=yaml.SafeLoader)

# Or skip the above nonsense and hard-code your token below
api = tator.get_api(**config)

# Set the following variables accordingly
project_id = 31
state_type = 149
media_type = 55
single_media_type = 53
algorithm_name = "ResNet 50 Feature Extraction"  # Double check this

# Define the vessels
vessels = {}
vessels["Arctic Fury (AF)"] = {"tag": "AF-"}
vessels["Lisa Melinda (LM)"] = {"tag": "LM-"}
vessels["Noahs Ark (NA)"] = {"tag": "NA-"}

vessel_tag_name_map = {}
for name in vessels:
    vessel_tag_name_map[vessels[name]["tag"]] = name

for vessel_name in vessels:
    this_vessel = vessels[vessel_name]
    this_vessel["sections"] = []

sections = api.get_section_list(project=project_id)
section_list = []
for section in sections:
    if "_Trip" in section.name:
        
        found_match = False
        for tag in vessel_tag_name_map:
            vessel_name = vessel_tag_name_map[tag]
            if tag in section.name:
                vessels[vessel_name]["sections"].append(section)
                section_list.append(section)
                found_match = True
                break
                
        if not found_match:
            print(f"Did not find matching vessel for {section.name}")
            
for name in vessels:
    print(f"{name} - {len(vessels[name]['sections'])} trips")


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

trip_keys = [key for key in trip_ids.keys()]

# Get 
all_media_ids_by_trip = {section.name: set() for section in section_list}
for sec in section_list:
    if list(trip_ids[sec.name].keys()) == []:
        states = []
    else:
        states = []
        for idx in range(0, len(trip_ids[sec.name].keys()), 100):
            states_chunk = api.get_state_list(
                project=project_id,
                media_id=list(trip_ids[sec.name].keys())[idx:idx+100],
                type=state_type,
            )
            states += states_chunk
    print(f"Section: {sec.name}")
    print(f"Number of ids: {len(trip_ids[sec.name].keys())}")
    print(f"Num States: {len(states)}")
    all_media_ids_by_trip[sec.name].update(mid for state in states for mid in state.media)

pruned_media_ids_by_trip = {section.name: [] for section in section_list}
for trip, media_id_set in all_media_ids_by_trip.items():
    start = 0
    step = 250
    media_id_list = list(media_id_set)

    while start < len(media_id_list):
        id_sublist = media_id_list[start : start + step]
        pruned_media_ids_by_trip[trip].extend(
            m.id for m in api.get_media_list(project_id, media_id=id_sublist, type=single_media_type)
        )
        start += step

pruned_media_ids = [
    ele for media_id_set in pruned_media_ids_by_trip.values() for ele in media_id_set
]

print(pruned_media_ids)
# Uncomment to save to yaml file for inspection
###############################################
#with open("test.yaml", "w") as fp:
#    yaml.dump(pruned_media_ids_by_trip, fp)

# Uncomment to launch the algorithm
###################################
# algorithm_launch_spec = {"algorithm_name": algorithm_name, "media_ids": pruned_media_ids}
# response = api.algorithm_launch(project=project_id, algorithm_launch_spec=algorithm_launch_spec)
# print(response)
