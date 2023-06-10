"""Tool used to take detections from a media and create tracks using IoU and KCF
"""

import argparse
import datetime
import logging
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import scipy.optimize

import tator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def _safe_retry(function, *args,**kwargs):
    count = kwargs.get('_retry_count', 5)
    if '_retry_count' in kwargs:
        del kwargs['_retry_count']
    complete = False
    fail_count = 0
    while complete is False:
        try:
            return_value = function(*args,**kwargs)
            complete = True
            return return_value
        except Exception as e:
            fail_count += 1
            if fail_count > count:
                raise e

class FrameBuffer():

    def __init__(
            self,
            tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
            media: tator.openapi.tator_openapi.models.media.Media,
            moving_forward: bool,
            work_folder: str,
            buffer_size: int,
            use_get_frame: bool=False) -> None:
        """ Constructor
        """

        self.tator_api = tator_api
        self.media_num_frames = media.num_frames - 2 #TODO Need to revisit once 2 frame bug is gone
        self.moving_forward = moving_forward
        self.buffer_size = buffer_size
        self.work_folder = work_folder
        self.media = media

        # Frames will be indexed by frame number. Each entry will be the 3 channel np matrix
        # that can be directly used by opencv, etc.
        self.frame_buffer = {}

        # Utilize the GetFrame endpoint for every request if asked to do so
        self.use_get_frame = use_get_frame

        # Determine highest resolution and related codec
        highest_quality = 0
        for media_file in media.media_files.streaming:
            if media_file.resolution[0] > highest_quality:
                self.hq_media_file = media_file

    def get_frame(self, frame: int) -> np.ndarray:
        """ Returns image to process from cv2.imread
        """

        if self.use_get_frame:
            temp_path = self.tator_api.get_frame(id=self.media.id, frames=[frame])
            image = cv2.imread(temp_path)
            os.remove(temp_path)
            return image

        else:

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
        logger.info(f"_refresh_frame_buffer -- start_frame: {start_frame}, last_frame: {last_frame}")

        clip = self.tator_api.get_clip(
            self.media.id,
            frame_ranges=[f"{start_frame}:{last_frame}"])
        temporary_file = clip.file
        save_path =  os.path.join(self.work_folder, temporary_file.name)
        for progress in tator.util.download_temporary_file(self.tator_api,
                                                           temporary_file,
                                                           save_path):
            continue

        # Re-create clip with h264 if av1 codec.
        # #TODO Revisit this once opencv is built with av1 compatibility
        if self.hq_media_file.codec == "av1":
            transcoded_file = os.path.join(self.work_folder, f"av1_converted.mp4")
            cmd = [
                "ffmpeg",
                "-i", save_path,
                "-vcodec", "libx264",
                "-crf", f"23",
                transcoded_file
            ]        
            subprocess.run(cmd)
            os.remove(save_path)
            save_path = transcoded_file

        # Create a list of frame numbers associated with the video clip
        # We will assume the clip returned encompasses this range
        frame_list = list(range(start_frame, last_frame + 1))

        # With the video downloaded, process the video and save the imagery into the buffer
        self.frame_buffer = {}
        reader = cv2.VideoCapture(save_path)
        while reader.isOpened():
            ok, frame = reader.read()
            if not ok:
                break
            if len(frame_list) == 0:
                break
            self.frame_buffer[frame_list.pop(0)] = frame.copy()
        reader.release()
        os.remove(save_path)

        end_time = time.time()

def extend_track(
        tator_api: tator.api,
        media_id: int,
        state_id: int,
        start_localization_id: int,
        direction: str,
        work_folder: str,
        max_coast_frames: int=0,
        max_extend_frames: int=5) -> None:
    """ Extends the track using the given track's detection using a visual tracker

    :param tator_api: Connection to Tator REST API
    :param media_id: Media ID associated with the track
    :param state_id: State/track ID to extend
    :param start_localization_id: Localization/detection to start the track extension with.
    :param direction: 'forward'|'backward'
    :param max_coast_frames: Number of coasted frames allowed if the tracker fails to
                             track in the given frame.

    This function will ignore existing detections.

    The track extension will stop once the maximum number of coast frames has been hit
    or if the start/end of the video havs been reached.

    """

    # Make sure the provided direction makes sense
    if direction.lower() == 'forward':
        moving_forward = True
    elif direction.lower() == 'backward':
        moving_forward = False
    else:
        raise ValueError("Invalid direction provided.")

    # Initialize the visual tracker with the start detection
    media = tator_api.get_media(id=media_id, _request_timeout=60*5)

    # Frame buffer that handles grabbing images from the video
    frame_buffer = FrameBuffer(
        tator_api=tator_api,
        media_id=media.id,
        media_num_frames=media.num_frames,
        moving_forward=moving_forward,
        work_folder=work_folder,
        buffer_size=300)

    start_detection = tator_api.get_localization(id=start_localization_id, _request_timeout=60*5)
    current_frame = start_detection.frame
    image = frame_buffer.get_frame(frame=current_frame)
    media_width = image.shape[1]
    media_height = image.shape[0]

    roi = [
        start_detection.x * media_width,
        start_detection.y * media_height,
        start_detection.width * media_width,
        start_detection.height * media_height]

    # Test, expand the ROI before feeding it into the tracker. This might yield
    # in better tracking
    roi[0] = int(max(0, roi[0] - roi[2]*0.0))
    roi[1] = int(max(0, roi[1] - roi[3]*0.0))
    roi[2] = int(min(image.shape[1], roi[2] + roi[2]*0.0))
    roi[3] = int(min(image.shape[0], roi[3] + roi[3]*0.0))
    roi = tuple(roi)

    tracker = cv2.TrackerKCF_create()
    ret = tracker.init(image, roi)

    # If the tracker fails to create for some reason, then bounce out of this routine.
    if not ret:
        log_msg = f'Tracker init failed. '
        logger.warning(log_msg)
        return
    else:
        previous_roi = roi
        previous_roi_image = image.copy()

    # Loop over the frames and attempt to continually track
    coasting = False
    coast_frames = 0
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
        image = frame_buffer.get_frame(frame=current_frame)

        if coasting:
            # Track coasted the last frame. Re-create the visual tracker using
            # the last known good track result before attempting to track this frame.
            logging.info("...coasted")
            tracker = cv2.TrackerKCF_create()
            ret = tracker.init(previous_roi_image, previous_roi)

            if not ret:
                break

        # Run the tracker with the current frame image
        ret, roi = tracker.update(image)

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
            coast_frames = coast_frames + 1 if coasting else 1
            coasting = True

        # If the maximum number of coast frames is reached, we're done
        # trying to track.
        if coasting and coast_frames >= max_coast_frames:
            break

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
            type=start_detection.type,
            frame=det.frame,
            x=x,
            y=y,
            width=width,
            height=height)

        localizations.append(detection_spec)

    created_ids = []
    for response in tator.util.chunked_create(
            tator_api.create_localization_list,
            media.project,
            body=localizations):
        created_ids += response.id

    tator_api.update_state(id=state_id, state_update={'localization_ids_add': created_ids}, _request_timeout=60*5)

def calculate_iou(
        boxA: tuple,
        boxB: tuple) -> float:
    """ Computes intersection over union for two bounding boxes.

    :param boxA: First box. Must be an array of form [x, y, w, h].
    :param boxB: Second box. Must be an array of form [x, y, w, h].

    :return: Intersection over union.

    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(int(boxA[0]), int(boxB[0]))
    yA = max(int(boxA[1]), int(boxB[1]))
    xB = min(int(boxA[0]) + int(boxA[2]), int(boxB[0]) + int(boxB[2]))
    yB = min(int(boxA[1]) + int(boxA[3]), int(boxB[1]) + int(boxB[3]))

    # compute the area of intersection rectangle
    interX = xB - xA + 1
    interY = yB - yA + 1
    if interX < 0 or interY < 0:
        iou = 0.0
    else:
        interArea = float((xB - xA + 1) * (yB - yA + 1))
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = int(boxA[2]) * int(boxA[3])
        boxBArea = int(boxB[2]) * int(boxB[3])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if float(boxAArea + boxBArea - interArea) <= 0.0:
            return 0.01
        iou = interArea / float(boxAArea + boxBArea - interArea)

        if iou > 1.0: # Sometimes get floating point precision errors
            iou = 1.0

    return iou

class Detection():

    def __init__(
            self,
            x: int,
            y: int,
            width: int,
            height: int,
            frame: int,
            det_id: int,
            confidence: float):
        """ Constructor
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.frame = frame
        self.id = det_id
        self.confidence = confidence

class Track():

    # Class wide variable used to generate unique track IDs
    new_track_id = 0

    def __init__(self, detection: Detection, max_coast_age:int) -> None:
        """ Constructor
        """

        # Unique ID of this track. Mostly useful for debugging purposes
        self.track_id = Track.new_track_id

        # List of detections in this track. Detection per frame and
        # the last entry is the most recently associated / generated (from coast).
        self.detection_list = [detection]

        # If a track extension is performed, detections related to that will appear in this list.
        # It's important it's in this list because there are assumptions the detection_list
        # is generated moving forward.
        self.back_extend_detection_list = []

        # Will be created in the update() function and used when the track is coasting
        self.tracker = None

        # Used to determine if the tracker needs to be replaced.
        self.last_tracker_frame = -2

        # Useful
        self.age = 0

        # Coast age increments when no detection is associated with the track.
        # Coast age rests to 0 otherwise.
        self.coast_age = 0

        # Set the maximum coast age
        self.max_coast_age = max_coast_age

        # Increment the class wide track ID so that we will always have a unique ID
        # when using this constructor
        Track.new_track_id += 1

        # Keep track of the start and end detections. If there are duplicates in a given frame,
        # it'll just be one of the detections in the frame.
        self.start_detection_list_index = 0
        self.start_detection_frame = detection.frame
        self.last_detection_list_index = 0
        self.last_detection_frame = detection.frame

    def associate_detection(self, detection: Detection) -> None:
        """ Associates the given detection to this track
        """
        self.detection_list.append(detection)

        if self.start_detection_frame > detection.frame:
            self.start_detection_list_index = len(self.detection_list) - 1
            self.start_detection_frame = detection.frame

        if self.last_detection_frame < detection.frame:
            self.last_detection_list_index = len(self.detection_list) - 1
            self.last_detection_frame = detection.frame

    def score_detection_association(self, detection: Detection) -> float:
        """ Returns an association score of 1.0 to 0.0 between the detection and track

        Best association score is 1.0 and worst is 0.0
        Utilizes an IoU comparison to determine if the detection associates.
        If the track is already considered dead, this function will return 0.0.

        """

        # Track dead? don't associate
        if self.is_dead():
            return 0.0

        # Detection already associated with this track this frame? don't associate
        if self.detection_list[-1].frame == detection.frame:
            return 0.0

        # Generate an association score using the IoU
        track_pos = self.detection_list[-1] # Grab last frame's detection for the IoU comparison
        boxA = [track_pos.x, track_pos.y, track_pos.width, track_pos.height]
        boxB = [detection.x, detection.y, detection.width, detection.height]

        iou = calculate_iou(boxA=boxA, boxB=boxB)
        return iou

    def is_dead(self) -> bool:
        """ Returns true if the track is dead (ie coasted for too long)
        """

        return self.coast_age >= self.max_coast_age

    def extend_backwards(
            self,
            frame: int,
            frame_buffer: dict) -> None:
        """
        """
        current_image = frame_buffer[frame]
        last_detection = self.detection_list[-1]

        tracker = cv2.TrackerKCF_create()
        roi = [last_detection.x, last_detection.y, last_detection.width, last_detection.height]
        roi[0] = int(max(0, roi[0] - roi[2]*0.0))
        roi[1] = int(max(0, roi[1] - roi[3]*0.0))
        roi[2] = int(min(current_image.shape[1], roi[2] + roi[2]*0.0))
        roi[3] = int(min(current_image.shape[0], roi[3] + roi[3]*0.0))
        tracker.init(current_image, tuple(roi))

        current_frame = frame
        if current_frame == 0:
            return

        last_frame = max(0, frame - self.max_coast_age)
        done = False
        while not done:
            current_frame = current_frame - 1
            current_image = frame_buffer[current_frame]
            ret, roi = tracker.update(current_image)

            if ret:
                # First verify the dimensions are ok / make sense
                image_width = current_image.shape[1]
                image_height = current_image.shape[0]
                #logger.info(f"Frame: Extend backwards - track {self.track_id} image (w,h) {image_width} {image_height} roi (x,y,w,h) {roi} frame {current_frame}")
                x = roi[0]
                y = roi[1]
                width = roi[2]
                height = roi[3]
                dimensions_ok = width > 0 and height > 0 and width <= image_width and height <= image_height and x < image_width and y < image_height

                if dimensions_ok:
                    # Create the new detection for this coasted frame and add it
                    # to the detection list
                    x = 0 if x < 0 else x
                    y = 0 if y < 0 else y
                    new_detection = Detection(
                        frame=current_frame,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        det_id=None,
                        confidence=None)
                    self.back_extend_detection_list.append(new_detection)

            else:
                done = True

            if current_frame == last_frame:
                done = True

    def finalize(self):
        """ Called when a track is done (e.g. being promoted from tracklet to track)
        """

        self.detection_list.extend(self.back_extend_detection_list)

    def update(
            self,
            frame: int,
            current_image: np.ndarray,
            previous_image: np.ndarray) -> None:
        """ Call this at the end of processing a frame

        If a track is coasted this past frame, increment the coast age and use the tracker
            to create a detection for this track in this frame.

        If not, only need to the reset coast age to 0

        """

        self.age += 1

        # Track coast this frame?
        last_detection = self.detection_list[-1]
        coasted = last_detection.frame != frame
        if coasted:

            # Track coasted. Use the tracker to create the detection.
            self.coast_age += 1

            if self.last_tracker_frame != frame - 1:
                # Didn't use the tracker last frame, so we got to create a new one
                # and initialize it with the last frame.
                self.tracker = cv2.TrackerKCF_create()
                roi = [last_detection.x, last_detection.y, last_detection.width, last_detection.height]

                # Test, expand the ROI before feeding it into the tracker. This might yield
                # in better tracking
                roi[0] = int(max(0, roi[0] - roi[2]*0))
                roi[1] = int(max(0, roi[1] - roi[3]*0))
                roi[2] = int(min(previous_image.shape[1], roi[2] + roi[2]*0))
                roi[3] = int(min(previous_image.shape[0], roi[3] + roi[3]*0))

                self.tracker.init(previous_image, tuple(roi))

            # Update the tracker with the current image and see if tracking is successful
            ret, roi = self.tracker.update(current_image)

            create_new_detection = False

            if ret:
                # First verify the dimensions are ok / make sense
                image_width = current_image.shape[1]
                image_height = current_image.shape[0]
                #logger.info(f"Frame: {frame} tracker update - track {self.track_id} image (w,h) {image_width} {image_height} roi (x,y,w,h) {roi}")
                x = roi[0]
                y = roi[1]
                width = roi[2]
                height = roi[3]
                dimensions_ok = width > 0 and height > 0 and width <= image_width and height <= image_height and x < image_width and y < image_height

                if dimensions_ok:
                    # Create the new detection for this coasted frame and add it
                    # to the detection list
                    x = 0 if x < 0 else x
                    y = 0 if y < 0 else y

                    new_detection = Detection(
                        frame=frame,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        det_id=None,
                        confidence=None)

                    self.associate_detection(new_detection)
                    create_new_detection = True

            if not create_new_detection:
                # Uh oh, tracker failed with the update. This track is now dead.
                self.coast_age = self.max_coast_age

            self.last_tracker_frame = frame

        else:
            # Track didn't coast, just reset the coast age.
            self.coast_age = 0

class TrackManager():

    def __init__(
            self,
            track_class,
            association_score_threshold: float,
            max_coast_age: int) -> None:
        """ Constructor
        """

        # List of tracks that detections can associate to at the current frame
        self.tracklets = []

        # Tracklets that are finalized and are tracks
        self.final_track_list = []

        # Image of previous frame to use when the tracker needs to be initialized
        self.previous_frame = None

        # Track type to use
        self.TrackClass = track_class

        # Maximum coast age for each tracklet
        self.max_coast_age = max_coast_age

        # Threshold for passing association scores
        self.association_score_threshold = association_score_threshold

        # Keep track of all the new states we made in tator
        self.state_id_list = []

    def process_detections(
            self,
            frame: int,
            detection_list: list) -> None:
        """ Process the list of detections for the current frame
        """

        # If there are no detections, don't bother
        num_dets = len(detection_list)
        if num_dets == 0:
            return

        # Any tracklets?
        num_tracklets = len(self.tracklets)
        if num_tracklets > 0:
            # Yep, first create an association matrix between detections and tracklets
            association_matrix = np.zeros((num_dets, num_tracklets))

            for det_idx, detection in enumerate(detection_list):
                for tracklet_idx, track in enumerate(self.tracklets):
                    score = track.score_detection_association(detection=detection)
                    association_matrix[det_idx, tracklet_idx] = score

            # Next, use the hungarian method to match up detections wtih tracks.
            # Since this method minimizes the weights between two sets, we will flip around the
            # association score (which normally 0.0 indicates a no match, 1.0 indicates an exact match)
            # Also grab the unassociated detections (e.g. scores don't pass, less tracks than detections)
            det_assignments, tracklet_assignments = scipy.optimize.linear_sum_assignment(1.0 - association_matrix)
            all_det_ids = np.arange(num_dets)
            unassociated_det_ids = list(np.setdiff1d(all_det_ids, det_assignments))

            for det_id, tracklet_id in zip(det_assignments, tracklet_assignments):
                score = association_matrix[det_id, tracklet_id]
                if score > self.association_score_threshold:
                    self.tracklets[tracklet_id].associate_detection(detection=detection_list[det_id])
                else:
                    unassociated_det_ids.append(det_id)

            # For each unassigned detection, create a new tracklet
            for det_id in unassociated_det_ids:
                self.tracklets.append(
                    self.TrackClass(
                        detection=detection_list[det_id],
                        max_coast_age=self.max_coast_age))

        else:
            # No, make each detection a new tracklet
            for detection in detection_list:
                self.tracklets.append(
                    self.TrackClass(
                        detection=detection,
                        max_coast_age=self.max_coast_age))

    def process_end_of_frame(
            self,
            frame: int,
            frame_buffer: dict,
            extend_tracks: bool) -> None:
        """ Performs required operations at the end of frame.

        Expected to be called after the detections have been processed.

        """

        current_image = frame_buffer[frame]

        previous_frame = frame - 1
        if previous_frame in frame_buffer:
            previous_image = frame_buffer[previous_frame]
        else:
            previous_image = None

        for tracklet in self.tracklets:

            # #TODO May want to parameterize this
            #if extend_tracks and tracklet.age == 0:
            #    #logger.info(f"[Frame {frame}] - Extend tracklets backwards")
            #    tracklet.extend_backwards(frame=frame, frame_buffer=frame_buffer)

            tracklet.update(
                frame=frame,
                current_image=current_image,
                previous_image=previous_image)

            if tracklet.is_dead():
                tracklet.finalize()
                self.final_track_list.append(tracklet)

        self.tracklets = [tracklet for tracklet in self.tracklets if not tracklet.is_dead()]

    def promote_tracklets_to_tracks(self) -> None:
        """ Forces moving all current tracklets to the final track list
        """
        for tracklet in self.tracklets:
            tracklet.finalize()
            self.final_track_list.append(tracklet)

        self.tracklets = []

    def save_tracks_to_tator(
            self,
            tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
            media: tator.openapi.tator_openapi.models.media.Media,
            image_width: int,
            image_height: int,
            min_num_detections: int,
            min_total_confidence: int,
            detection_type_id: int,
            state_type_id: int,
            track_version: int,
            create_tator_entries: None) -> None:
        """ Saves the tracks in final_track_list to tator. Those tracks are removed internally.
        """

        for track in self.final_track_list:

            # Loop over the detections and grab the IDs to associate with the new track
            # If needed, create a localization for coasted detections
            detection_ids = []
            dets = [] # new detections
            localization_list = [] # with scaled x/y
            total_confidence = 0.0
            num_detections = 0
            for det in track.detection_list:

                # Create localization spec
                width = image_width - det.x if det.x + det.width > image_width else det.width
                height = image_height - det.y if det.y + det.height > image_height else det.height
                x = 0.0 if det.x < 0 else det.x / image_width
                y = 0.0 if det.y < 0 else det.y / image_height
                width = width / image_width
                height = height / image_height

                if det.id is None:
                    # Have to create a detection for this coasted frame
                    detection_spec = dict(
                        media_id=media.id,
                        type=detection_type_id,
                        frame=det.frame,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        version=track_version)
                    dets.append(detection_spec)
                else:
                    num_detections += 1
                    detection_ids.append(det.id)

                    # Undo the detection scaling
                    scale_det_factor = 1.5
                    delta_width = width - width / scale_det_factor
                    delta_height = height - height / scale_det_factor
                    width = width / scale_det_factor
                    height = height / scale_det_factor
                    x += delta_width / 2
                    y += delta_height / 2

                    # Note: This really should be just part of the else statement, since everything that
                    #       doesn't have an ID should be a detection with a confidence associated.
                    #       But, we will defensively program here.
                    try:
                        total_confidence += float(det.confidence)
                    except:
                        pass

                loc = SimpleNamespace(
                    id=det.id,
                    media_id=media.id,
                    type=detection_type_id,
                    frame=det.frame,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    version=track_version)
                localization_list.append(loc)

            # If this track did not meet the minimum detection / confidence thresholds, then ignore
            # this track and move to the next track
            if num_detections < min_num_detections or total_confidence < min_total_confidence:
                continue

            if create_tator_entries is not None:
                create_tator_entries({
                    "detection_list": localization_list,
                    "new_detections": dets})

            else:
                # Bulk create the new detections
                for idx in range(0, len(dets), 500):
                    response = tator_api.create_localization_list(
                        project=media.project,
                        body=dets[idx:idx+500],
                        _request_timeout=60)
                    detection_ids += response.id

                # Create track
                state_spec = dict(
                    type=state_type_id,
                    frame=track.detection_list[0].frame,
                    version=track_version,
                    localization_ids=detection_ids,
                    media_ids=[media.id],
                    attributes={})

                # #TODO
                state_spec["attributes"]["NOAA Label"] = "-"
                state_spec["attributes"]["Label"] = "-"

                response = tator_api.create_state_list(project=media.project, body=[state_spec], _request_timeout=60*5)
                self.state_id_list.append(response.id)

        self.final_track_list = []

def email_status(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        project: int,
        subject: str,
        message: str) -> None:
    """ Send an email to the executor of this script
    """

    user = tator_api.whoami()
    email_spec = {
      'recipients': [user.email],
      'subject': subject,
      'text': message,
      'attachments': [],
    }
    response = tator_api.send_email(project=project, email_spec=email_spec, _request_timeout=60*5)

def process_media(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        media_id: int,
        max_coast_age: int,
        association_threshold: int,
        min_num_detections: int,
        min_total_confidence: float,
        detection_type_id: int,
        state_type_id: int,
        detection_version: int,
        track_version: int,
        local_video_file_path: str='',
        extend_track: bool=False,
        start_frame: int=-1,
        create_tator_entries=None,
        email_subject: str=None) -> None:
    """ Process single media
    """

    media = tator_api.get_media(id=media_id, _request_timeout=60*5)

    # Gather all the detections in the given media
    detections = tator_api.get_localization_list(
        project=media.project,
        media_id=[media.id],
        type=detection_type_id,
        version=[detection_version],
        _request_timeout=60*10)

    # If there are no detections, then just get out of here.
    if len(detections) == 0:
        msg = "No detections in media"
        logger.info(msg)
        return
    else:
        logger.info(f"Processing {len(detections)} detections")

    # Make the detections accessible by frame
    min_frame = media.num_frames
    max_frame = 0
    detection_frame_assoc = {}
    for det in detections:

        if det.frame not in detection_frame_assoc:
            detection_frame_assoc[det.frame] = [det]
            if det.frame < min_frame and det.frame >= start_frame:
                min_frame = det.frame

            if det.frame > max_frame:
                max_frame = det.frame

        else:
            detection_frame_assoc[det.frame].append(det)

    # If the video file is local, use that instead of querying images through tator
    video_reader = None
    if os.path.exists(local_video_file_path):
        video_reader = cv2.VideoCapture(local_video_file_path)

    # Now, cycle through the frames. Start with the first frame there is a detection
    # to the last detection frame + the max coast age or the end of the media.
    # Perform tracking with the detections as we cycle through the frames.

    logger.info(f"Tracking parameters (max_coast_age, association_threshold, extend_track): {max_coast_age} {association_threshold} {extend_track}")

    track_mgr = TrackManager(
        track_class=Track,
        association_score_threshold=association_threshold,
        max_coast_age=max_coast_age)

    start_frame = max(0, min_frame - max_coast_age)
    end_frame = min(media.num_frames, max_frame + max_coast_age)
    current_image = None
    frame_buffer = {}
    logger.info(f"Processing frames: {start_frame} to {end_frame}")

    image_width = None
    image_height = None

    if video_reader is not None:
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    else:
        media = tator_api.get_media(id=media_id)
        frame_buffer_interface = FrameBuffer(
            tator_api=tator_api,
            media=media,
            moving_forward=True,
            work_folder="/tmp", #TODO This may need to change
            buffer_size=200,
            use_get_frame=False)

    start_time = datetime.datetime.now()
    for frame in range(start_frame, end_frame):

        if frame % 1000 == 0:
            end_time = datetime.datetime.now()
            msg = f"Processing frame: {frame} ({end_time - start_time} seconds)"
            logger.info(msg)
            start_time = datetime.datetime.now()

        if frame % 50000 == 0:
            if email_subject is not None:
                email_message = f"[{datetime.datetime.now()}] still tracking - at frame number: {frame} (last det frame: {end_frame}) ({frame/media.num_frames:.2f}%)"
                email_status(tator_api=tator_api, project=media.project, subject=email_subject, message=email_message)

        if video_reader is None:
            current_image = frame_buffer_interface.get_frame(frame)

        else:
            ok, current_image = video_reader.read()
            if not ok:
                break

        if image_height is None:
            image_height = current_image.shape[0]
            image_width = current_image.shape[1]
            logger.info(f"Image dimensions (height/width): {image_height} {image_width}")

        frame_buffer[frame] = current_image
        if len(frame_buffer) > max_coast_age + 2:
            del frame_buffer[frame - max_coast_age - 2]

        # Grab the detections list for this frame. We need to convert the dimensions
        # to integer pixel values
        orig_det_list = detection_frame_assoc[frame] if frame in detection_frame_assoc else []
        det_list = []
        for det in orig_det_list:

            scale_det_factor = 1.5
            orig_width = det.width * image_width
            orig_height = det.height * image_height
            width = orig_width * scale_det_factor
            width = width if width < image_width else image_width
            height = orig_height * scale_det_factor
            height = height if height < image_height else image_height
            delta_width = width - orig_width
            delta_height = height - orig_height
            x = det.x * image_width - delta_width / scale_det_factor
            x = x if x >= 0 else 0
            y = det.y * image_height - delta_height / scale_det_factor
            y = y if y >= 0 else 0

            det_list.append(
                Detection(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    frame=frame,
                    det_id=det.id,
                    confidence=det.attributes["Confidence"]))

        track_mgr.process_detections(
            frame=frame,
            detection_list=det_list)

        track_mgr.process_end_of_frame(
            frame=frame,
            frame_buffer=frame_buffer,
            extend_tracks=extend_track) # #TODO Do we need this?

        track_mgr.save_tracks_to_tator(
            tator_api=tator_api,
            media=media,
            image_width=image_width,
            image_height=image_height,
            min_num_detections=min_num_detections,
            min_total_confidence=min_total_confidence,
            detection_type_id=detection_type_id,
            state_type_id=state_type_id,
            track_version=track_version,
            create_tator_entries=create_tator_entries)

    # Finalize the remaining track list if there is any
    track_mgr.promote_tracklets_to_tracks()
    track_mgr.save_tracks_to_tator(
        tator_api=tator_api,
        media=media,
        image_width=image_width,
        image_height=image_height,
        min_num_detections=min_num_detections,
        min_total_confidence=min_total_confidence,
        detection_type_id=detection_type_id,
        state_type_id=state_type_id,
        track_version=track_version,
        create_tator_entries=create_tator_entries)

    #_safe_retry(tator_api.update_media, id=media.id, media_update={"attributes": {"_OO_track_frames_processed": frame}}, _request_timeout=60*1)

def parse_args():
    """ Get the arguments passed into this script.

    Utilizes tator's parser which has its own based added arguments

    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--host", type=str, default="https://cloud.tator.io")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--media-id", type=int, required=True)
    parser.add_argument("--local-video-file", type=str, default="")
    parser.add_argument('--max-coast-age', type=int, help='Maximum track coast age', default=5)
    parser.add_argument('--association-threshold', type=float, help='Passing association threshold', default=0.1)
    parser.add_argument('--min-num-detections', type=int, help='Minimum number of detections (not generated by tracker) for a valid track', default=1)
    parser.add_argument('--min-total-confidence', type=float, help='Minimum total sum of detection confidences for a valid track', default=-1.0)
    parser.add_argument('--track-type', type=int)
    parser.add_argument('--detection-type', type=int)
    parser.add_argument('--detection-version', type=int)
    parser.add_argument('--track-version', type=int)
    parser.add_argument('--extend-track', action="store_true")
    parser.add_argument("--start-frame", type=int)
    return parser.parse_args()

def main():
    """ Main function of this script
    """

    # Parse arguments and set up API.
    args = parse_args()

    tator_api = tator.get_api(host=args.host, token=args.token)
    process_media(
        tator_api=tator_api,
        media_id=args.media_id,
        local_video_file_path=args.local_video_file,
        max_coast_age=args.max_coast_age,
        association_threshold=args.association_threshold,
        min_num_detections=args.min_num_detections,
        min_total_confidence=args.min_total_confidence,
        detection_type_id=args.detection_type,
        state_type_id=args.track_type,
        detection_version=args.detection_version,
        track_version=args.track_version,
        extend_track=args.extend_track,
        start_frame=args.start_frame)

if __name__ == '__main__':
    main()