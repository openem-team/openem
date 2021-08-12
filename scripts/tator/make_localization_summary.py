"""
"""
import argparse
import datetime
import json
import os
import shutil
import textwrap
import traceback
import types
import urllib
import zipfile

import pandas as pd
import progressbar
import tator

def email_status(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        project: int,
        subject: str,
        message: str,
        attachments: list=[]) -> None:
    """
    """
    user = tator_api.whoami()
    email_spec = {
      'recipients': [user.email],
      'subject': subject,
      'text': message,
      'attachments': attachments,
    }
    response = tator_api.send_email(project=project, email_spec=email_spec)

def convert_filter_for_tator_search(filter_condition: dict) -> str:
    """ Converts the given filter condition into a Tator REST compliant search string
    """

    modifier_str = filter_condition["modifier"]
    modifier_end_str = ""
    if filter_condition["modifier"] == "==":
        modifier_str = ""
    elif filter_condition["modifier"] == "Includes":
        modifier_str = "*"
        modifier_end_str = "*"

    # Lucene search string requires spaces and parentheses to have a preceding backslash
    field_str = filter_condition["field"].replace(" ","\\ ").replace("(","\\(").replace(")","\\)")
    value_str = filter_condition["value"].replace(" ","\\ ").replace("(","\\(").replace(")","\\)")

    search = f"{field_str}:{modifier_str}{value_str}{modifier_end_str}"
    return search

def create_search_string(filter_conditions: list) -> str:
    """
    :param filter_conditions: List of dicts with the following keys:
        field
        modifier
        value
    """
    search = ""
    for idx, filter_condition in enumerate(filter_conditions):
        search += convert_filter_for_tator_search(filter_condition)
        if idx < len(filter_conditions) - 1:
            search += " AND "

    return search

def parse_filters_for_localizations(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        project: int,
        json_filters: list) -> tuple:
    """
    :param json_filters: List of filters in JSON form

    #TODO States are currently not supported
    """

    # Get the types
    media_types = tator_api.get_media_type_list(project=project)
    media_types_names = [media_type.name for media_type in media_types]
    localization_types = tator_api.get_localization_type_list(project=project)
    localization_types_names = [loc_type.name for loc_type in localization_types]

    # Get the sections
    sections = tator_api.get_section_list(project=project)

    # Separate the types out into their groups
    media_filter_conditions = []
    localization_filter_conditions = []
    version_ids = []
    dtype_ids = []
    media_ids = []
    for filter_condition in json_filters:
        if filter_condition["category"] in media_types_names:
            if filter_condition["field"] == "_section":
                section_id = int(filter_condition["value"].split("(ID:")[1].replace(")",""))
                for section in sections:
                    if section.id == section_id:
                        new_filter_condition = {
                            "field": "tator_user_sections",
                            "modifier": "==",
                            "value": section.tator_user_sections
                        }
                        media_filter_conditions.append(new_filter_condition)
                        break
            elif filter_condition["field"] == "_id":
                media_ids.append(int(filter_condition["value"]))
            else:
                media_filter_conditions.append(filter_condition)

        elif filter_condition["category"] in localization_types_names:
            if filter_condition["field"] == "_version":
                version = int(filter_condition["value"].split("(ID:")[1].replace(")",""))
                version_ids.append(version)
            elif filter_condition["field"] == "_dtype":
                dtype = int(filter_condition["value"].split("(ID:")[1].replace(")",""))
                dtype_ids.append(dtype)
            else:
                localization_filter_conditions.append(filter_condition)

    # Create the respective search strings
    media_search = create_search_string(filter_conditions=media_filter_conditions)
    localization_search = create_search_string(filter_conditions=localization_filter_conditions)

    return localization_search, media_search, media_ids, version_ids, dtype_ids

def create_files(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        project: int,
        localizations: list,
        work_dir: str,
        total_image_size_threshold_gb: float) -> tuple:
    """
    """

    # Get the project's sections
    sections = tator_api.get_section_list(project=project)

    # Get the media associated with the localizations
    media_ids = set()
    for localization in localizations:
        media_ids.add(localization.media)
    medias = tator_api.get_media_list_by_id(project=project, media_id_query={"ids": list(media_ids)})
    media_map = {media.id: media for media in medias}

    # Get the localization type information
    localization_types = tator_api.get_localization_type_list(project=project)
    localization_type_map = {loc_type.id: loc_type for loc_type in localization_types}

    # Get the users associated with this project
    user_map = {}
    bar = progressbar.ProgressBar()
    report_data = []
    image_file_names = []
    image_file_size_gb = 0
    bytes_to_gb = 1.0 / (1024 * 1024 * 1024)
    bytes_to_mb = 1.0 / (1024 * 1024)
    disable_image_downloads = False
    for localization in bar(localizations):

        # Store the media
        media = media_map[localization.media]

        # Grab the URL associated with this localization
        url = f"{tator_api.api_client.configuration.host}/{project}/annotation/{media.id}?frame={localization.frame}"
        url += f"&selected_entity={localization.id}"
        url += f"&selected_type={localization.meta}"
        if localization.version is not None:
            url += f"&version={localization.version}"

        # Get the associated user object if it's not already cached
        if localization.user not in user_map:
            user_map[localization.user] = tator_api.get_user(localization.user)

        loc_type = localization_type_map[localization.meta]

        datum={
            'id': localization.id,
            "media_id": media.id,
            "media": media.name,
            'frame': localization.frame,
            "type_name": loc_type.name,
            "type_dtype": loc_type.dtype,
            "type_id": loc_type.id,
            'x': localization.x * media.width if localization.x is not None else None,
            'y': localization.y * media.height if localization.y is not None else None,
            "width": 0.0,
            "height": 0.0,
            'user': user_map[localization.user].username,
            'user_id': localization.user,
            'url': url,
            'thumbnail': "",
            "thumbnail_size_mb": 0}

        datum.update(localization.attributes)

        # Apply the width/height attributes based on the localization dtype
        if loc_type.dtype == 'box':
            datum["width"] = localization.width * media.width if localization.width is not None else None
            datum["height"] = localization.height * media.height if localization.height is not None else None

        elif loc_type.dtype == 'line':
            datum["width"] = localization.u * media.width if localization.u is not None else None
            datum["height"] = localization.v * media.height if localization.v is not None else None

        # Save the attributes of the localization specific to the localization type
        datum.update(localization.attributes)
        report_data.append(datum)

        # Download the image
        margin_x = 0
        margin_y = 0
        if loc_type.dtype == "dot":
            margin_x = 50
            margin_y = 50
            loc_x_pixels = int(localization.x * width)
            loc_y_pixels = int(localization.y * height)

            if loc_y_pixels - margin_y < 0:
                margin_y = loc_y_pixels

            elif loc_y_pixels + margin_y > height:
                margin_y = height - loc_y_pixels

            if margin_x < minimum_margin or margin_y < minimum_margin:
                msg = f"Dot graphic of {localization.id} not retrieved. Margins too small (x,y margins: {margin_x} {margin_y})"
                print(msg)
                continue

        retry_count = 0
        image_path = None
        if not disable_image_downloads:
            while retry_count >= 0:
                try:
                    image_path = tator_api.get_localization_graphic(
                        localization.id,
                        use_default_margins=False,
                        margin_x=margin_x,
                        margin_y=margin_y)
                    break
                except:
                    retry_count -= 1

        if image_path is None:
            if not disable_image_downloads:
                print(f"Could not download graphic for localization {localization.id}")
        else:
            final_img_name = f"{localization.id}.png"
            target_path = os.path.join(work_dir, final_img_name)
            shutil.move(image_path, target_path)
            image_file_names.append(target_path)
            image_file_size_gb += os.path.getsize(target_path) * bytes_to_gb
            if image_file_size_gb > total_image_size_threshold_gb:
                disable_image_downloads = True
            datum["thumbnail"] = final_img_name
            datum["thumbnail_size_mb"] = os.path.getsize(target_path) * bytes_to_mb

    # Get the attributes associated with the localizations in this project
    # and save them as a report column
    # Create the CSV file
    column_names=[
        'media_id','media','id', 'user', 'user_id','frame','type_name','type_id','type_dtype',
        'x','y','width','height','url','thumbnail','thumbnail_size_mb']

    for loc_type in localization_types:
        for attr in loc_type.attribute_types:
            if attr.name not in column_names:
                column_names.append(attr.name)

    df = pd.DataFrame(data=report_data, columns=column_names)
    file_name = os.path.join(work_dir, "localization_summary.csv")
    df.to_csv(file_name, index=False)

    return file_name, image_file_names

def main(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        project: int,
        encoded_filters: str,
        job_id: str,
        work_dir: str="",
        no_email: bool=False,
        email_subject: str="Localization Summary",
        total_image_size_threshold_gb: float=2.0) -> None:
    """
    """

    # Parse out the filters that is a JSON encoded string. We do need the double decode here.
    json_string = urllib.parse.unquote(urllib.parse.unquote(encoded_filters))
    print(f"Decoded JSON string: {json_string}")

    json_filters = json.loads(json_string)
    print(f"Parsed JSON: {json_filters}")

    localization_search, media_search, media_ids, version_ids, dtype_ids = \
        parse_filters_for_localizations(
            tator_api=tator_api,
            project=project,
            json_filters=json_filters)

    print("Parsed filters:")
    print(f"media_search: {media_search}")
    print(f"localization_search: {localization_search}")
    print(f"media_ids: {media_ids}")
    print(f"version: {version_ids}")
    print(f"dtypes: {dtype_ids}")

    function_args = {"project": project}
    if media_search != "":
        function_args["media_search"] = media_search
    if localization_search != "":
        function_args["search"] = localization_search
    if len(version_ids) > 0:
        function_args["version"] = version_ids
    if len(media_ids) > 0:
        function_args["media_id"] = media_ids

    print("")
    print(f"Localization endpoint arguments:")
    print(function_args)

    localizations = []
    if len(dtype_ids) > 0:
        for dtype in dtype_ids:
            count = tator_api.get_localization_count(
                **function_args,
                type=dtype)
            print(f"Retrieving {count} localizations (type={dtype})")

            current_localizations = []
            page_size = 9000
            after = None
            while len(current_localizations) < count:
                if after is None:
                    current_localizations.extend(tator_api.get_localization_list(**function_args, type=dtype, start=0, stop=page_size))
                    after = current_localizations[-1].id
                else:
                    current_localizations.extend(tator_api.get_localization_list(**function_args, type=dtype, start=0, stop=page_size, after=after))
                    after = current_localizations[-1].id
            localizations.extend(current_localizations)

    else:
        count = tator_api.get_localization_count(**function_args)
        print(f"Retrieving {count} localizations")

        current_localizations = []
        page_size = 9000
        after = None
        while len(current_localizations) < count:
            if after is None:
                current_localizations.extend(tator_api.get_localization_list(**function_args, start=0, stop=page_size))
                after = current_localizations[-1].id
            else:
                current_localizations.extend(tator_api.get_localization_list(**function_args, start=0, stop=page_size, after=after))
                after = current_localizations[-1].id
        localizations.extend(current_localizations)

    # Organize the filter information into an easy to read string
    # category:field modifier value
    filter_conditions_string = ""
    for filter_condition in json_filters:
        filter_conditions_string += f'-    {filter_condition["category"]}:{filter_condition["field"]} {filter_condition["modifier"]} {filter_condition["value"]}\n'

    # Get the project name for the email message
    project_obj = tator_api.get_project(id=project)

    if not no_email:
        email_status(
            tator_api=tator_api,
            project=project,
            subject=email_subject,
            message=textwrap.dedent(
f"""Workflow launched to create report of {count} localizations from {project_obj.name} - (ID: {project})

The following filters were applied:
{filter_conditions_string}

An email with the same subject line will be sent upon report completion.

Tator Workflow Job ID: {job_id}

"""))

    print(f"Creating .csv file and extracting images with {total_image_size_threshold_gb}Gb threshold")
    csv_file_path, image_file_paths = create_files(
        tator_api=tator_api,
        project=project,
        localizations=localizations,
        work_dir=work_dir,
        total_image_size_threshold_gb=total_image_size_threshold_gb)

    # Rest of this function is email related. If it was requested to not email the results,
    # then we won't bother with uploading to Tator and sending out an email with links + logs
    if no_email:
        return

    # Get image information
    image_info_str = f"Number of images extracted: {len(image_file_paths)} of {len(localizations)}\n"
    if len(image_file_paths) != len(localizations):
        image_info_str += f"-   Missing {len(localizations) - len(image_file_paths)} images\n"

        total_size_gb = 0
        bytes_to_gb = 1.0 / (1024 * 1024 * 1024)
        for image_file_path in image_file_paths:
            total_size_gb += os.path.getsize(image_file_path) * bytes_to_gb
        if total_size_gb > total_image_size_threshold_gb:
            image_info_str += f"-   Exceeded {total_image_size_threshold_gb}Gb threshold. Apply additional filter constraints to reduce the localization count."

    # Package up the image files into it's own zip file and remove the images files
    img_zip_path = None
    if len(image_file_paths) > 0:
        img_zip_path = os.path.join(work_dir, "localization_summary_images.zip")
        print(f"Zipping image files into: {img_zip_path}")
        with zipfile.ZipFile(img_zip_path, 'w') as zip_handle:
            for img_path in image_file_paths:
                zip_handle.write(img_path)
                os.remove(img_path)

    # Package up everything into a zip file
    zip_out_path = os.path.join(work_dir, "localization_summary.zip")
    with zipfile.ZipFile(zip_out_path, 'w') as zip_handle:
        print(f"Zipping {csv_file_path} into {zip_out_path}")
        zip_handle.write(csv_file_path)
        if os.path.exists(img_zip_path):
            print(f"Zipping {img_zip_path} into {zip_out_path}")
            zip_handle.write(img_zip_path)
            os.remove(img_zip_path)

    # Upload the .csv file to Tator as a temporary file
    print(f"Uploading temporary file: {zip_out_path}")
    for progress, response in tator.util.upload_temporary_file(
            api=tator_api,
            project=project,
            path=zip_out_path):
        pass
    temp_file_id = response.id
    zip_temp_file = tator_api.get_temporary_file(id=temp_file_id)

    # Create the e-mail
    email_message = textwrap.dedent(
f"""Report of {count} localizations created from {project_obj.name} - (ID: {project}). The .zip file below contains the report.

The following filters were applied:
{filter_conditions_string}

{image_info_str}

Download the .zip file using the following Tator link. Please note this link will expire in approximately 24 hours ({zip_temp_file.eol_datetime})

Link: {zip_temp_file.path}

Tator Workflow Job ID: {job_id}

""")

    # Send out the email
    email_status(
        tator_api=tator_api,
        project=project,
        subject=email_subject,
        message=email_message)

def parse_workflow_env_vars() -> types.SimpleNamespace:
    """ Get environment variables used during workflow mode
    """
    env_vars = types.SimpleNamespace()
    env_vars.host = os.getenv('TATOR_API_SERVICE')
    if env_vars.host is not None:
        env_vars.host = env_vars.host.rsplit('/', 1)[0]
    env_vars.token = os.getenv('TATOR_AUTH_TOKEN')
    env_vars.project = os.getenv('TATOR_PROJECT_ID')
    env_vars.work_dir = os.getenv('TATOR_WORK_DIR')
    return env_vars

def parse_args() -> argparse.Namespace:
    """ Get the script arguments
    """

    parser = argparse.ArgumentParser(description="Uploads the classifier results to Tator")
    parser.add_argument("--host", type=str, default="https://www.tatorapp.com")
    parser.add_argument("--token", type=str)
    parser.add_argument("--project", type=int)
    parser.add_argument("--uid", type=str)
    parser.add_argument("--work-dir", type=str, default="./")
    parser.add_argument("--encoded-filters", type=str, default="%5B%5D")
    parser.add_argument("--total-image-size-threshold-gb", type=float, default=0.5)
    parser.add_argument("--no-email", action="store_true", help="Prevent email of results")
    return parser.parse_args()

def script_main() -> None:
    """ Module entry point
    """

    print("")
    print("Running make_annotation_summary")
    print("")

    env_vars = parse_workflow_env_vars()
    args = parse_args()

    # See if we're running in workflow mode. If so, grab data from there
    if env_vars.host is not None:
        # Workflow parameters
        project = env_vars.project
        token = env_vars.token
        host = env_vars.host
        work_dir = env_vars.work_dir

    else:
        # Script parameters
        project = args.project
        token = args.token
        host = args.host
        work_dir = args.work_dir

    print(env_vars)
    print(args)
    print("\n")

    os.makedirs(work_dir, exist_ok=True)
    print(f"Working directory: {work_dir}\n")

    tator_api = tator.get_api(host=host, token=token)
    jobs = tator_api.get_job_list(project=project)
    job_id = None
    for job in jobs:
        if job.uid == args.uid:
            job_id = job.id
            break

    project_obj = tator_api.get_project(id=project)
    current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    email_subject = f"Filtered Localization Summary - {project_obj.name} [{current_time}]"

    try:
        main(
            tator_api=tator_api,
            project=project,
            encoded_filters=args.encoded_filters,
            job_id=job_id,
            no_email=args.no_email,
            email_subject=email_subject,
            work_dir=work_dir,
            total_image_size_threshold_gb=args.total_image_size_threshold_gb)

    except Exception as exc:
        print(exc)
        print(f"{traceback.format_exc()}")

        if not args.no_email:
            email_status(
                tator_api=tator_api,
                project=project,
                subject=email_subject,
                message=textwrap.dedent(f"""
Error occurred. Please contact CVision AI.

Arguments:
- uid: {args.uid}
- job_id: {job_id}
- host: {host}
- project: {project}
- work_dir: {work_dir}
- total-image-size-threshold-gb: {args.total_image_size_threshold_gb}
- encoded-filters: {args.encoded_filters}

Traceback error:
{traceback.format_exc()}
                """))

if __name__ == "__main__":
    script_main()