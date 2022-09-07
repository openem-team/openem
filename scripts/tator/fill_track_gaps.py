"""

This Tator utility script performs visual tracking for a specific track to either extend
a track from a specific spot, or fill in gaps of empty frames.

"""

import argparse
import logging
import multiprocessing
import os
import threading
import time
import sys
import urllib.parse
from typing import List
from types import SimpleNamespace

import numpy as np
import cv2

import tator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameBuffer():

    def __init__(
            self,
            tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
            media_id: int,
            media_num_frames: int,
            moving_forward: bool,
            work_folder: str,
            buffer_size: int) -> None:
        """ Constructor
        """

        self.tator_api = tator_api
        self.media_id = media_id
        self.media_num_frames = media_num_frames - 2 #TODO Need to revisit once 2 frame bug is gone
        self.moving_forward = moving_forward
        self.buffer_size = buffer_size
        self.work_folder = work_folder

        # Frames will be indexed by frame number. Each entry will be the 3 channel np matrix
        # that can be directly used by opencv, etc.
        self.frame_buffer = {}

    def get_frame(self, frame: int) -> np.ndarray:
        """ Returns image to process from cv2.imread
        """

        # Have we already read the frame we care about?
        if frame not in self.frame_buffer:

            # Nope, looks like we need to refresh the buffer.
            # If we are moving backwards in the media, then we should jump further back.
            start_frame = frame
            if not self.moving_forward:
                start_frame = frame - self.buffer_size
                start_frame = 0 if start_frame < 0 else start_frame

            self._refresh_frame_buffer(start_frame=start_frame)

            # Check again, if frame is still not in the frame buffer after refreshing,
            # we've got a problem. And bounce out.
            if frame not in self.frame_buffer:
                raise ValueError("Problem refreshing frame buffer")

        return self.frame_buffer[frame]

    def _refresh_frame_buffer(
            self,
            start_frame: int) -> None:
        """ Refreshes the frame buffer by getting a video clip and reading the video's frames

        Postcondition(s):
            self.frame_buffer is set with numpy arrays, indexed by frame number
        """

        start_time = time.time()

        # Request the video clip and download it
        last_frame = start_frame + self.buffer_size
        last_frame = self.media_num_frames if last_frame > self.media_num_frames else last_frame
        clip = self.tator_api.get_clip(
            self.media_id,
            frame_ranges=[f"{start_frame}:{last_frame}"])
        temporary_file = clip.file
        save_path =  os.path.join(self.work_folder, temporary_file.name)
        for progress in tator.util.download_temporary_file(self.tator_api,
                                                           temporary_file,
                                                           save_path):
            continue

        # Create a list of frame numbers associated with the video clip
        frame_list = []
        for start_frame, end_frame in zip(clip.segment_start_frames, clip.segment_end_frames):
            frame_list.extend(list(range(start_frame, end_frame + 1)))

        # With the video downloaded, process the video and save the imagery into the buffer
        self.frame_buffer = {}
        reader = cv2.VideoCapture(save_path)
        while reader.isOpened():
            ok, frame = reader.read()
            if not ok:
                break
            self.frame_buffer[frame_list.pop(0)] = frame.copy()
        reader.release()
        os.remove(save_path)

        end_time = time.time()

def extend_track(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        media_id: int,
        state_id: int,
        start_localization_id: int,
        direction: str,
        work_folder: str,
        max_coast_frames: int=0,
        max_extend_frames: int=None) -> None:
    """ Extends the track using the given track's detection using a visual tracker

    :param tator_api: Connection to Tator REST API
    :param media_id: Media ID associated with the track
    :param state_id: State/track ID to extend
    :param start_localization_id: Localization/detection to start the track extension with.
        The attributes of this detection will be copied over to subsequent detections
        created during the extension process.
    :param direction: 'forward'|'backward'
    :param work_folder: Folder that will contain the images
    :param max_coast_frames: Number of coasted frames allowed if the tracker fails to
                             track in the given frame.
    :param max_extend_frames: Maximum number of frames to extend. Track extension will stop if
                              coasting occurs still or if the start/end of the video has been
                            reached.

    This function will ignore existing detections.

    """

    logger.info(f"media_id: {media_id}")
    logger.info(f"state_id: {media_id}")
    logger.info(f"max_coast_frames: {max_coast_frames}")
    logger.info(f"max_extend_frames: {max_extend_frames}")
    logger.info(f"direction: {direction}")

    # Make sure the provided direction makes sense
    if direction.lower() == 'forward':
        moving_forward = True
    elif direction.lower() == 'backward':
        moving_forward = False
    else:
        raise ValueError("Invalid direction provided.")

    # Initialize the visual tracker with the start detection
    media = tator_api.get_media(id=media_id)

    # Frame buffer that handles grabbing images from the video
    frame_buffer = FrameBuffer(
        tator_api=tator_api,
        media_id=media.id,
        media_num_frames=media.num_frames,
        moving_forward=moving_forward,
        work_folder=work_folder,
        buffer_size=200)
    logger.info("Frame buffer initialized")

    start_detection = tator_api.get_localization(id=start_localization_id)
    current_frame = start_detection.frame
    image = frame_buffer.get_frame(frame=current_frame)
    media_width = image.shape[1]
    media_height = image.shape[0]

    logger.info(f"media (width, height) {media_width} {media_height}")
    logger.info(f"start_detection: {start_detection}")

    roi = (
        start_detection.x * media_width,
        start_detection.y * media_height,
        start_detection.width * media_width,
        start_detection.height * media_height)

    tracker = cv2.legacy.TrackerCSRT_create()
    ret = tracker.init(image, roi)

    # If the tracker fails to create for some reason, then bounce out of this routine.
    if not ret:
        log_msg = f'Tracker init failed. '
        log_msg += f'Localization: {start_detection} State: Media: {media}'
        logger.warning(log_msg)
        return
    else:
        previous_roi = roi
        previous_roi_image = image.copy()
    logger.info("Tracker initialized")

    # Loop over the frames and attempt to continually track
    coasting = False
    new_detections = []
    frame_count = 0

    while True:

        # For now, only process the a certain amount of frames
        if frame_count == max_extend_frames:
            break
        frame_count += 1

        # Get the frame number in the right extension direction
        current_frame = current_frame + 1 if moving_forward else current_frame - 1

        # Stop processing if the tracker is operating outside of the valid frame range
        if current_frame < 0 or current_frame >= media.num_frames - 2:
            break

        # Grab the image
        start = time.time()
        image = frame_buffer.get_frame(frame=current_frame)
        end = time.time()

        start = time.time()
        if coasting:
            # Track coasted the last frame. Re-create the visual tracker using
            # the last known good track result before attempting to track this frame.
            logging.info("...coasted")
            tracker = cv2.legacy.TrackerCSRT_create()
            ret = tracker.init(previous_roi_image, previous_roi)

            if not ret:
                break

        # Run the tracker with the current frame image
        ret, roi = tracker.update(image)
        end = time.time()

        if ret:
            # Tracker was successful, save off the new detection position/time info
            # Also save off the image in-case the tracker coasts the next frame
            coasting = False
            previous_roi = roi
            previous_roi_image = image.copy()

            new_detections.append(
                SimpleNamespace(
                    frame=current_frame,
                    x=roi[0],
                    y=roi[1],
                    width=roi[2],
                    height=roi[3]))

        else:
            # Tracker was not successful and the track is coasting now.
            # If the maximum number of coast frames is reached, we're done
            # trying to track.
            coast_frames = coast_frames + 1 if coasting else 1
            coasting = True

            if coast_frames == max_coast_frames:
                break

    # Alter the attributes. If there's a None, put in a ""
    # Otherwise, there will be an error when attempting to create these new localizations
    attributes = start_detection.attributes.copy()
    for key in attributes:
        if attributes[key] == None:
            attributes[key] = ""

    # Finally, create the new localizations and add them to the state
    localizations = []
    for det in new_detections:

        x = 0.0 if det.x < 0 else det.x / media_width
        y = 0.0 if det.y < 0 else det.y / media_height

        width = media_width - det.x if det.x + det.width > media_width else det.width
        height = media_height - det.y if det.y + det.height > media_height else det.height
        width = width / media_width
        height = height / media_height

        detection_spec = dict(
            media_id=start_detection.media,
            type=start_detection.meta,
            frame=det.frame,
            x=x,
            y=y,
            width=width,
            height=height,
            version=start_detection.version,
            **attributes)

        localizations.append(detection_spec)

    # These are encapsulated in try/catch blocks to delete newly created localizations
    # if something goes awry
    created_ids = []
    try:
        for response in tator.util.chunked_create(
                tator_api.create_localization_list,
                media.project,
                localization_spec=localizations):
            created_ids += response.id

    except:
        for loc_id in created_ids:
            tator_api.delete_localization(id=loc_id)
        created_ids = []
        raise ValueError("Problem creating new localizations")

    try:
        if len(created_ids) > 0:
            tator_api.update_state(id=state_id, state_update={'localization_ids_add': created_ids})

    except:
        for loc_id in created_ids:
            tator_api.delete_localization(id=loc_id)
        raise ValueError("Problem updating state with new localizations")

def calc_interpolation_params(
    det1 : tator.models.LocalizationSpec,
    det2 : tator.models.LocalizationSpec) -> List:
    """ Calculates interpolation params between detections
    """
    frame_1 = det1.frame
    frame_2 = det2.frame
    frame_delta = frame_2 - frame_1
    mx = (det2.x + det2.width/2) - (det1.x + det1.width/2)
    my = (det2.y + det2.height/2) - (det1.y + det1.height/2)
    mw = det2.width - det1.width
    mh = det2.height - det1.height

    return frame_delta, mx/frame_delta, my/frame_delta, mw/frame_delta, mh/frame_delta

def linearly_interpolate_sparse_track(
    tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        media_id: int,
        state_id: int) -> None:
    """ Fills in gaps of detections for the given track

    :param tator_api: Connection to Tator REST API
    :param media_id: Media ID associated with the track
    :param state_id: State/track ID to extend

    This checks for the existence of a track with sparse detections. For every pair
    of detections, it will calculate a linear interpolation of size and position
    between the pair, and fill in the gaps.

    """

    media = tator_api.get_media(id=media_id)
    track = tator_api.get_state(id=state_id)
    detections = {}
    detection_frames = []
    for detection_id in track.localizations:
        det = tator_api.get_localization(id=detection_id)
        if det.frame not in detections:
            detections[det.frame] = [det]
            detection_frames.append(det.frame)

    detection_frames.sort()
    detection_pairs = []
    # Find consecutive dets with gaps
    for i,frame in enumerate(sorted(detection_frames)):
        if i + 1 >= len(detection_frames):
            break
        if detection_frames[i+1] - frame > 1:
            detection_pairs.append((frame, detection_frames[i+1]))

    
    # Calc inerpolation params for pairs
    interpolation_params = []
    for det_pair in detection_pairs:
        det1 = detections.get(det_pair[0])[0]
        det2 = detections.get(det_pair[1])[0]
        det_pair_params = calc_interpolation_params(det1, det2)
        interpolation_params.append(det_pair_params)

    # Create interpolated detections for pairs
    for det_pair, params in zip (detection_pairs,interpolation_params):
        frame_start = det_pair[0] + 1
        frame_delta, mx, my, mw, mh = params
        det_start = detections.get(frame_start-1)[0]

        localizations = []
        for frame in range(frame_delta - 1):
            x = 0.0 if det.x < 0 else det_start.x + det_start.width/2 + mx*(frame+1) - (det_start.width + mw*(frame+1))/2
            y = 0.0 if det.y < 0 else det_start.y + det_start.height/2 + my*(frame+1) - (det_start.height + mh*(frame+1))/2

            width = 1.0 - x if det.x + det.width > 1.0 else det_start.width + mw*(frame+1)
            height = 1.0 - y if det.y + det.height > 1.0 else det_start.height + mh*(frame+1)

            detection_spec = dict(
                media_id=media_id,
                type=det_start.meta,
                frame=frame_start + frame,
                x=x,
                y=y,
                width=width,
                height=height,
                version=det_start.version,
                **det_start.attributes)

            localizations.append(detection_spec)

        # These are encapsulated in try/catch blocks to delete newly created localizations
        # if something goes awry
        created_ids = []
        try:
            for response in tator.util.chunked_create(
                    tator_api.create_localization_list,
                    media.project,
                    localization_spec=localizations):
                created_ids += response.id

        except:
            for loc_id in created_ids:
                tator_api.delete_localization(id=loc_id)
            created_ids = []
            raise ValueError("Problem creating new localizations")

        try:
            if len(created_ids) > 0:
                tator_api.update_state(id=state_id, state_update={'localization_ids_add': created_ids})

        except:
            for loc_id in created_ids:
                tator_api.delete_localization(id=loc_id)
            raise ValueError("Problem updating state with new localizations")
    
def fill_sparse_track(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        media_id: int,
        state_id: int,
        work_folder: str) -> None:
    """ Fills in gaps of detections for the given track

    :param tator_api: Connection to Tator REST API
    :param media_id: Media ID associated with the track
    :param state_id: State/track ID to extend
    :param work_folder: Folder that will contain the images

    This loops through each of the frames of the tracks. For frames where a
    detection/localization does not exist, it'll attempt to use a visual tracker
    from the original list of detections.

    If there are multiple detections with a frame, currently it'll only use
    one of the detections for the key frame / initializing the tracker.

    If the visual tracker coasts, it'll just duplicate the last localization made.

    Currently, this algorithm doesn't reset the tracker when it starts coasting
    within the gaps. #TODO Worth revisting this.

    """

    # Initialize the visual tracker with the start detection
    media = tator_api.get_media(id=media_id)

    # Frame buffer that handles grabbing images from the video
    frame_buffer = FrameBuffer(
        tator_api=tator_api,
        media_id=media.id,
        media_num_frames=media.num_frames,
        moving_forward=True,
        work_folder=work_folder,
        buffer_size=200)

    # Grab the detections in the given track
    track = tator_api.get_state(id=state_id)
    detections = {}
    for detection_id in track.localizations:
        det = tator_api.get_localization(id=detection_id)
        if det.frame not in detections:
            detections[det.frame] = [det]
        else:
            detections[det.frame].append(det)

    # Grab the frame range we will be operating on based on the existing detections
    sorted_det_frames = list(detections.keys())
    sorted_det_frames.sort()
    frames = list(range(sorted_det_frames[0], sorted_det_frames[-1] + 1))

    # We will copy over the attributes of the starting detection to the new detections
    # that fill in the gaps
    start_detection = detections[sorted_det_frames[0]][0]

    # Loop over the frame range and fill in the gaps if there are any
    new_detections = []
    for frame in frames:

        if frame not in detections:

            # If we are already coasting, don't bother right now
            # trying to re-create the tracker.
            if not coasted:
                if tracker is None:

                    # Grab the image
                    start = time.time()
                    image = frame_buffer.get_frame(frame=last_detection.frame)
                    end = time.time()

                    media_width = image.shape[1]
                    media_height = image.shape[0]

                    roi = (
                        last_detection.x * media_width,
                        last_detection.y * media_height,
                        last_detection.width * media_width,
                        last_detection.height * media_height)
                    last_roi = roi

                    tracker = cv2.legacy.TrackerCSRT_create()
                    ret = tracker.init(image, roi)

                    if not ret:
                        # Problem creating the tracker. Just duplicate the last detection
                        logger.warning("Error creating tracker.")
                        coasted = True
                        new_detections.append(
                            SimpleNamespace(
                                frame=frame,
                                x=roi[0],
                                y=roi[1],
                                width=roi[2],
                                height=roi[3]))

                        continue


                # Run the tracker with the current frame image
                start = time.time()
                image = frame_buffer.get_frame(frame=frame)
                end = time.time()
                ret, roi = tracker.update(image)

                # If the tracker failed, use the last good ROI
                if not ret:
                    coasted = True
                    new_detections.append(
                        SimpleNamespace(
                            frame=frame,
                            x=last_roi[0],
                            y=last_roi[1],
                            width=last_roi[2],
                            height=last_roi[3]))

                else:
                    new_detections.append(
                        SimpleNamespace(
                            frame=frame,
                            x=roi[0],
                            y=roi[1],
                            width=roi[2],
                            height=roi[3]))
                    last_roi = roi

            else:
                new_detections.append(
                    SimpleNamespace(
                        frame=frame,
                        x=last_roi[0],
                        y=last_roi[1],
                        width=last_roi[2],
                        height=last_roi[3]))

        else:
            # New keyframe, clear the visual tracker.
            last_detection = detections[frame][0]
            tracker = None
            coasted = False

    # Finally, create the new localizations and add them to the state
    localizations = []
    for det in new_detections:

        x = 0.0 if det.x < 0 else det.x / media_width
        y = 0.0 if det.y < 0 else det.y / media_height

        width = media_width - det.x if det.x + det.width > media_width else det.width
        height = media_height - det.y if det.y + det.height > media_height else det.height
        width = width / media_width
        height = height / media_height

        detection_spec = dict(
            media_id=start_detection.media,
            type=start_detection.meta,
            frame=det.frame,
            x=x,
            y=y,
            width=width,
            height=height,
            **start_detection.attributes)

        localizations.append(detection_spec)

    # These are encapsulated in try/catch blocks to delete newly created localizations
    # if something goes awry
    created_ids = []
    try:
        for response in tator.util.chunked_create(
                tator_api.create_localization_list,
                media.project,
                localization_spec=localizations):
            created_ids += response.id

    except:
        for loc_id in created_ids:
            tator_api.delete_localization(id=loc_id)
        created_ids = []
        raise ValueError("Problem creating new localizations")

    try:
        if len(created_ids) > 0:
            tator_api.update_state(id=state_id, state_update={'localization_ids_add': created_ids})

    except:
        for loc_id in created_ids:
            tator_api.delete_localization(id=loc_id)
        raise ValueError("Problem updating state with new localizations")

def parse_args() -> argparse.Namespace:
    """ Returns the arguments passed to the script
    """
    parser = tator.get_parser()
    parser.add_argument('--url', type=str, help='URL to rest service.')
    parser.add_argument('--gid', type=str, default='', help='Group ID for sending progress.')
    parser.add_argument('--uid', type=str, default='', help='Job ID for sending progress.')
    parser.add_argument('--media', type=int, help='Media ID related to track.')
    parser.add_argument('--track', type=int, help='Track ID to process.')
    parser.add_argument('--algo', type=str, help='Algorithm to run. Options: fillgaps|extend|batchextend')
    parser.add_argument('--work-folder', type=str, default='/work', help='Work folder that clips will be downloaded to')
    parser.add_argument('--extend-direction', type=str, help='Extension algorithm direction. Options: forward|backward')
    parser.add_argument('--extend-detection-id', type=int, help='ID of detection to start the extension process with.')
    parser.add_argument('--extend-max-coast', type=int, default=0, help='Max coast frames to use with the extension algorithm.')
    parser.add_argument('--extend-max-frames', type=int, help="Max number of frames to extend if provided. If not provided, only coasting or end of media will prohibit the extension")
    parser.add_argument('--fill-strategy', type=str,help='Strategy for filling track gaps',
        default='visual')
    return parser.parse_args()

def main() -> None:
    """ Main function of this script
    """

    # Process the arguments
    args = parse_args()

    # Setup the interface to the tator server
    url = urllib.parse.urlparse(args.url)
    host = f"{url.scheme}://{url.netloc}"
    tator_api = tator.get_api(host=host, token=args.token)

    # Launch the algorithm depending on the provided arguments
    if args.algo == 'fillgaps':
        if args.fill_strategy == 'visual':
            fill_sparse_track(
                tator_api=tator_api,
                media_id=args.media,
                state_id=args.track,
                work_folder=args.work_folder)
        elif args.fill_strategy == 'linear':
            linearly_interpolate_sparse_track(
                tator_api=tator_api,
                media_id=args.media,
                state_id=args.track
            )
        else:
            raise ValueError(f"Invalid fill strategy provided: {args.fill_strategy}")

    elif args.algo == 'extend':
        extend_track(
            tator_api=tator_api,
            media_id=args.media,
            state_id=args.track,
            start_localization_id=args.extend_detection_id,
            direction=args.extend_direction,
            work_folder=args.work_folder,
            max_coast_frames=args.extend_max_coast,
            max_extend_frames=args.extend_max_frames)

    else:
        raise ValueError(f"Invalid algorithm provided: {args.algo}")

if __name__ == "__main__":
    main()
