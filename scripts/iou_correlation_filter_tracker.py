''' This utilizes a combination of IoU and a correlation filter to perform multiple object tracking
'''

import argparse
import logging
import os
import sys
import urllib.parse

import cv2
import numpy as np
import pandas as pd
import scipy.optimize

import tator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

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
            det_id: int):
        """ Constructor
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.frame = frame
        self.id = det_id

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

        # Will be created in the update() function and used when the track is coasting
        self.tracker = None

        # Used to determine if the tracker needs to be replaced.
        self.last_tracker_frame = -2

        # Coast age increments when no detection is associated with the track.
        # Coast age rests to 0 otherwise.
        self.coast_age = 0

        # Set the maximum coast age
        self.max_coast_age = max_coast_age

        # Increment the class wide track ID so that we will always have a unique ID
        # when using this constructor
        Track.new_track_id += 1

    def associate_detection(self, detection: Detection) -> None:
        """ Associates the given detection to this track
        """
        self.detection_list.append(detection)

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
        #boxA = (
        #    np.max((0,int(boxA[0] - boxA[2]*0.1))),
        #    np.max((0,int(boxA[1] - boxA[3]*0.1))),
        #    int(boxA[2] + boxA[2]*0.2),
        #    int(boxA[3] + boxA[3]*0.2))

        boxB = [detection.x, detection.y, detection.width, detection.height]
        #boxB = (
        #    np.max((0,int(boxB[0] - boxB[2]*0.05))),
        #    np.max((0,int(boxB[1] - boxB[3]*0.05))),
        #    int(boxB[2] + boxB[2]*0.05),
        #    int(boxB[3] + boxB[3]*0.05))

        iou = calculate_iou(boxA=boxA, boxB=boxB)
        return iou

    def is_dead(self) -> bool:
        """ Returns true if the track is dead (ie coasted for too long)
        """

        return self.coast_age >= self.max_coast_age

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

        # Track coast this frame?
        last_detection = self.detection_list[-1]
        coasted = last_detection.frame != frame
        if coasted:

            # Track coasted. Use the tracker to create the detection.
            self.coast_age += 1

            if self.last_tracker_frame != frame - 1:
                # Didn't use the tracker last frame, so we got to create a new one
                # and initialize it with the last frame.
                self.tracker = cv2.TrackerCSRT_create()
                roi = [last_detection.x, last_detection.y, last_detection.width, last_detection.height]
                self.tracker.init(previous_image, tuple(roi))

            # Update the tracker with the current image and see if tracking is successful
            ret, roi = self.tracker.update(current_image)

            create_new_detection = False

            if ret:
                # First verify the dimensions are ok / make sense
                image_width = current_image.shape[1]
                image_height = current_image.shape[0]
                logger.info(f"Frame: {frame} tracker update - image (w,h) {image_width} {image_height} roi (x,y,w,h) {roi}")
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
                        det_id=None)
                    self.detection_list.append(new_detection)
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
            current_image: np.ndarray,
            previous_image: np.ndarray) -> None:
        """ Performs required operations at the end of frame.

        Expected to be called after the detections have been processed.

        """

        for tracklet in self.tracklets:

            tracklet.update(
                frame=frame,
                current_image=current_image,
                previous_image=previous_image)

            if tracklet.is_dead():
                self.final_track_list.append(tracklet)
                self.tracklets.remove(tracklet)

    def promote_tracklets_to_tracks(self) -> None:
        """ Forces moving all current tracklets to the final track list
        """
        for tracklet in self.tracklets:
            self.final_track_list.append(tracklet)

        self.tracklets = []

def process_media(
        args,
        tator_api,
        media_id,
        local_video_file_path: str=''):
    """ Process single media
    """

    media = tator_api.get_media(id=media_id)

    # Grab the localization type that is a box. It's assumed that this project
    # has been set up to only have one localization box type (that will be the detections)
    detection_type_id = None
    box_type_counts = 0
    localization_types = tator_api.get_localization_type_list(project=media.project)
    for loc_type in localization_types:
        if loc_type.dtype == 'box':
            detection_type_id = loc_type.id
            box_type_counts += 1

    if detection_type_id is None:
        raise ValueError("No localization box type detected. Expected only one.")

    if box_type_counts > 1:
        raise ValueError("Multiple localization box types detected. Expected only one.")

    # Grab the state type. Assumed to only have one
    state_types = tator_api.get_state_type_list(project=media.project)
    state_type_id = state_types[0].id

    # Gather all the detections in the given media
    detections = tator_api.get_localization_list(project=media.project, media_id=[media.id])

    # If there are no detections, then just get out of here.
    if len(detections) == 0:
        msg = "No detections in media"
        logger.info(msg)
        return

    # Make the detections accessible by frame
    min_frame = media.num_frames
    max_frame = 0
    detection_frame_assoc = {}
    for det in detections:

        if det.frame not in detection_frame_assoc:
            detection_frame_assoc[det.frame] = [det]
            min_frame = det.frame if det.frame < min_frame else min_frame
            max_frame = det.frame if det.frame > max_frame else max_frame

        else:
            detection_frame_assoc[det.frame].append(det)

    # If the video file is local, use that instead of querying images through tator
    video_reader = None
    if os.path.exists(local_video_file_path):
        video_reader = cv2.VideoCapture(local_video_file_path)

    # Now, cycle through the frames. Start with the first frame there is a detection
    # to the last detection frame + the max coast age or the end of the media.
    # Perform tracking with the detections as we cycle through the frames.
    max_coast_age = args.max_coast_age
    passing_association_score_threshold = args.association_threshold
    track_mgr = TrackManager(
        track_class=Track,
        association_score_threshold=passing_association_score_threshold,
        max_coast_age=max_coast_age)

    start_frame = min_frame
    end_frame = min(media.num_frames, max_frame + max_coast_age)
    current_image = None
    previous_image = None
    for frame in range(start_frame, end_frame):

        msg = f'Processing frame: {frame}'
        logging.info(msg)

        previous_image = current_image

        if video_reader is None:
            image_path = tator_api.get_frame(id=media.id, frames=[frame])
            current_image = cv2.imread(image_path)
            os.remove(image_path)

        else:
            ok, current_image = video_reader.read()
            if not ok:
                raise ValueError(f"Problem with local video read of {local_video_file}")

        # Grab the detections list for this frame. We need to convert the dimensions
        # to integer pixel values
        orig_det_list = detection_frame_assoc[frame] if frame in detection_frame_assoc else []
        det_list = []
        for det in orig_det_list:
            det_list.append(
                Detection(
                    x=det.x * media.width,
                    y=det.y * media.height,
                    width=det.width * media.width,
                    height=det.height * media.height,
                    frame=frame,
                    det_id=det.id))
        track_mgr.process_detections(frame=frame, detection_list=det_list)

        track_mgr.process_end_of_frame(
            frame=frame,
            current_image=current_image,
            previous_image=previous_image)

    # Finalize the track list
    track_mgr.promote_tracklets_to_tracks()

    # Loop through each of the tracks.
    #   Create the coasted detections if there are any.
    #   Create the track
    for track in track_mgr.final_track_list:

        # Loop over the detections and grab the IDs to associate with the new track
        # If needed, create a localization for coasted detections
        detection_ids = []
        for det in track.detection_list:
            if det.id is None:
                # Have to create a detection for this coasted frame
                x = 0.0 if det.x < 0 else det.x / media.width
                y = 0.0 if det.y < 0 else det.y / media.height

                width = det.width - x if x + det.width > media.width else det.width
                height = det.height - y if y + det.height > media.height else det.height
                width = width / media.width
                height = height / media.height

                detection_spec = dict(
                    media_id=media.id,
                    type=detection_type_id,
                    frame=det.frame,
                    x=x,
                    y=y,
                    width=width,
                    height=height)

                response = tator_api.create_localization_list(
                    project=media.project,
                    localization_spec=[detection_spec])
                det.id = response.id[0]

            detection_ids.append(det.id)

        # Create track
        state_spec = dict(
            type=state_type_id,
            frame=track.detection_list[0].frame,
            localization_ids=detection_ids,
            media_ids=[media.id])
        tator_api.create_state_list(project=media.project, state_spec=[state_spec])

def parse_args():
    """ Get the arguments passed into this script.

    Utilizes tator's parser which has its own based added arguments

    """
    parser = tator.get_parser()
    parser.add_argument('--url', type=str, help='URL to rest service.')
    parser.add_argument('--media', type=int, help='Media ID of video to create tracks.')
    parser.add_argument('--csv', type=str, help='.csv file with local_media_file, media_id')
    parser.add_argument('--gid', type=str, help='Group Jobs ID')
    parser.add_argument('--uid', type=str, help='Job ID')
    parser.add_argument('--max-coast-age', type=int, help='Maximum track coast age', default=5)
    parser.add_argument('--association-threshold', type=float, help='Passing association threshold', default=0.4)
    return parser.parse_args()

def main():
    """ Main function of this script
    """

    # Parse arguments and set up API.
    args = parse_args()

    # Setup the interface to the tator server
    url = urllib.parse.urlparse(args.url)
    host = f"{url.scheme}://{url.netloc}"
    tator_api = tator.get_api(host=host, token=args.token)

    # Retrieve media information
    if args.media is not None:
        # Was provided a media ID to process
        process_media(
            args=args,
            tator_api=tator_api,
            media_id=args.media)

    else:
        # Was provided a .csv file that contains a list of local video files to process
        medias_df = pd.read_csv(args.csv)
        for row_idx, row in medias_df.iterrows():
            process_media(
                args=args,
                tator_api=tator_api,
                media_id=row['media_id'],
                local_video_file_path=row['media'])

if __name__ == '__main__':
    main()