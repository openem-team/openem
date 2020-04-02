import os
import platform
import pickle
import shutil
import glob
import json
import cv2
import tables
import numpy as np
from utilities import ensure_path_exists

class TrackData:
    """ Interface to directory containing track (training or inference)
        data.
    """
    def __init__(self, data_dir):
        """ Constructor.
            Inputs:
            data_dir -- Path to track data directory.
        """
        ## Base path to data directory.
        self.data_dir = data_dir.strip()

    def video_path(self):
        """ Returns path to video.
        """
        with open(os.path.join(self.data_dir, "video_path.txt"), "r") as f:
            return f.read().strip('\n')

    def save_video_path(self, video_path):
        """ Saves path to video used to create this track data.
        """
        with open(os.path.join(self.data_dir, "video_path.txt"), "w") as f:
            f.write(video_path)

    def video_size(self):
        """ Returns size of video.
        """
        vid = cv2.VideoCapture(self.video_path())
        vid.open(self.video_path())
        ok, img = vid.read()
        return img.shape

    def video_frame_rate(self):
        """ Returns frame rate of video.
        """
        vid = cv2.VideoCapture(self.video_path())
        return vid.get(cv2.CAP_PROP_FPS)

    def detections(self):
        """ Opens and reads detection file.  Returns unpickled file.
        """
        det_path = os.path.join(self.data_dir, "detections.pkl")
        det_file = open(det_path, "rb")
        return pickle.load(det_file)

    def save_detections(self, detections):
        """ Saves detections to file.
        """
        det_path = os.path.join(self.data_dir, "detections.pkl")
        det_file = open(det_path, "wb")
        pickle.dump(detections, det_file)

    def num_detection_images(self):
        """ Returns number of detection images.
        """
        det_dir = os.path.join(self.data_dir, "detection_images")
        if os.path.exists(det_dir):
            num_imgs = len(glob.glob(os.path.join(det_dir, "*.png")))
        else:
            num_imgs = 0
        return num_imgs

    def detection_image(self, index):
        """ Loads a detection image by index.
            index -- Index of requested detection image.
        """
        name = "det_img_{:09d}.png".format(index)
        det_dir = os.path.join(self.data_dir, "detection_images")
        path = os.path.join(det_dir, name)
        return cv2.imread(path).astype(np.float)

    def clear_detection_images(self):
        """ Clears detection images.
        """
        det_dir = os.path.join(self.data_dir, "detection_images")
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)

    def save_detection_image(self, det_img, index):
        """ Saves detection images.
            Inputs:
            det_imgs -- Detection images with consistent image size.
            index -- Index of this detection image.
        """
        name = "det_img_{:09d}.png".format(index)
        det_dir = os.path.join(self.data_dir, "detection_images")
        ensure_path_exists(det_dir)
        path = os.path.join(det_dir, name)
        cv2.imwrite(path, det_img)

    def detection_info(self):
        """ Returns detection image metadata.
        """
        path = os.path.join(self.data_dir, "detection_info.pkl")
        f = open(path, "rb")
        return pickle.load(f)

    def save_detection_info(self, det_info):
        """ Saves detection image metadata.
        """
        path = os.path.join(self.data_dir, "detection_info.pkl")
        f = open(path, "wb")
        pickle.dump(det_info, f)

    def clear_tracks(self):
        """ Deletes all current tracks.
        """
        track_dir = os.path.join(self.data_dir, "tracks")
        if os.path.exists(track_dir):
            shutil.rmtree(track_dir)

    def tracks(self):
        """ Opens and reads track files.
        """
        track_dir = os.path.join(self.data_dir, "tracks")
        trk_files = glob.glob(os.path.join(track_dir, "track*.pkl"))
        return [pickle.load(open(f, "rb")) for f in trk_files]

    def save_track(self, track):
        """ Saves track.
        """
        trk_fname = "track_{:04d}.pkl".format(track.id)
        trk_dir = os.path.join(self.data_dir, "tracks")
        ensure_path_exists(trk_dir)
        trk_path = os.path.join(trk_dir, trk_fname)
        trk_file = open(trk_path, "wb")
        pickle.dump(track, trk_file)

    def assoc_examples(self):
        """ Opens and reads association examples file.
        """
        assoc_path = os.path.join(self.data_dir, "assoc_examples.pkl")
        assoc_file = open(assoc_path, "rb")
        return pickle.load(assoc_file)

    def save_assoc_examples(self, assoc_examples):
        """ Saves association examples.
        """
        assoc_path = os.path.join(self.data_dir, "assoc_examples.pkl")
        assoc_file = open(assoc_path, "wb")
        pickle.dump(assoc_examples, assoc_file)

    def assoc_examples_all(self):
        """ Opens and reads all association examples file.
        """
        assoc_path = os.path.join(self.data_dir, "assoc_examples_all.pkl")
        assoc_file = open(assoc_path, "rb")
        return pickle.load(assoc_file)

    def save_assoc_examples_all(self, assoc_examples):
        """ Saves all association examples.
        """
        assoc_path = os.path.join(self.data_dir, "assoc_examples_all.pkl")
        assoc_file = open(assoc_path, "wb")
        pickle.dump(assoc_examples, assoc_file)

    def init_examples(self):
        """ Opens and reads track initiation examples file.
        """
        init_path = os.path.join(self.data_dir, "init_examples.pkl")
        init_file = open(init_path, "rb")
        return pickle.load(init_file)

    def save_init_examples(self, init_examples):
        """ Saves track initiation examples.
        """
        init_path = os.path.join(self.data_dir, "init_examples.pkl")
        init_file = open(init_path, "wb")
        pickle.dump(init_examples, init_file)

    def annotations(self):
        """ Opens and reads annotation file.
        """
        with open(os.path.join(self.data_dir, "tracks.json"), "r") as f:
            ann = json.load(f)
            return ann

    def save_annotations(self, annotations, iteration=None):
        """ Saves track annotation file.
        """
        if iteration:
            fname = 'tracks_{:04d}.json'.format(iteration)
        else:
            fname = 'tracks.json'
        with open(os.path.join(self.data_dir, fname), "w") as f:
            json.dump(annotations, f, indent=4, sort_keys=True)

    def save_annotations_by_video(self, annotations):
        """ Saves track annotation file alongside video with same basename.
        """
        vid_path = self.video_path()
        base, ext = os.path.splitext(vid_path)
        with open(base + ".json", "w") as f:
            json.dump(annotations, f, indent=4, sort_keys=True)

    def save_comments(self, comments):
        """ Appends a string to a text file containing comments.  If the
            comments file does not exist it will be created.
            Inputs:
            comments -- String containing comments.
        """
        path = os.path.join(self.data_dir, "comments.txt")
        if os.path.exists(path):
            f = open(path, "a")
        else:
            f = open(path, "w")
        f.write(comments)
        f.write("\n")
        f.close()

    def max_coast(self):
        """ Returns max coast that was used for this track data.
        """
        max_delta = 0
        for track in self.tracks():
            prev_frame = None
            for d in track.detections:
                this_frame = int(d['frame'])
                if not prev_frame is None:
                    delta = this_frame - prev_frame
                    max_delta = max(delta, max_delta)
                prev_frame = this_frame
        return max_delta

    def save_detections_to_json(self, detections, track_ids, iteration):
        """Saves detections labelled by track ID to json.
        
        # Arguments
            detections: List of detections.
            track_ids: ID of given detections.
            iteration: Suffix for json file.
        """
        assert(len(detections) == len(track_ids))
        written = []
        tracks = []
        dets = []
        for det, tid in zip(detections, track_ids):
            det["id"] = tid
            if not det["id"] in written:
                tracks.append({
                    "id" : str(det["id"]),
                    "species" : det["species"],
                    "subspecies" : "",
                    "frame_added" : det["frame"],
                    "count_label" : "ignore"})
                written.append(det["id"])
            dets.append({
                "id" : str(det["id"]),
                "x" : str(int(det["x"])),
                "y" : str(int(det["y"])),
                "w" : str(int(det["w"])),
                "h" : str(int(det["h"])),
                "species" : det["species"],
                "prob" : str(det["prob"]),
                "type" : "box",
                "frame" : str(det["frame"])})
        out = {
            "detections" : dets,
            "tracks" : tracks,
            "global_state" : ""}
        self.save_annotations(out, iteration)

    def save_pair_weights(self, pairs, weights, is_cut, iteration):
        """Saves pair weights to track directory.
       
        # Arguments 
            pairs: Tracklet ID pairs.
            weights: Weights computed for each pair.
            is_cut: Whether the edge was cut.
            iteration: Suffix for csv file.
        """
        assert(len(pairs) == len(weights))
        assert(len(pairs) == len(is_cut))
        fname = 'pair_weights_{:04d}.csv'.format(iteration)
        save_path = os.path.join(self.data_dir, fname)
        with open(save_path, 'w') as f:
            f.write("track_id0,track_id1,weight,is_cut\n")
            for (p0, p1), w, c in zip(pairs, weights, is_cut):
                f.write("{},{},{},{}\n".format(p0, p1, w, c))

    def save_constraints(self, constraints, iteration):
        """Saves constraint pairs to track directory.
        """
        fname = 'constraints_{:04d}.csv'.format(iteration)
        save_path = os.path.join(self.data_dir, fname)
        with open(save_path, 'w') as f:
            f.write("track_id0, track_id1\n")
            for (p0, p1) in constraints:
                f.write("{},{}\n".format(p0, p1))

    def clear_tracklets(self):
        path = os.path.join(self.data_dir, "tracklets.h5")
        if os.path.exists(path):
            os.remove(path)

    def num_tracklet_pairs(self):
        """Returns number of tracklet pairs.
        """
        path = os.path.join(self.data_dir, "tracklets.h5")
        f = tables.open_file(path, 'r')
        num_examples = 0
        for seq_len in f.root.pairs_same:
            num_examples += seq_len.shape[0]
        for seq_len in f.root.pairs_diff:
            num_examples += seq_len.shape[0]
        f.close()
        return num_examples

    def tracklets(self):
        """Returns tracklets as a pair of mem mapped arrays.

        # Arguments
            seq_len: Sequence length.

        # Returns
            Tuple containing appearance features, spatiotemporal features.
        """
        path = os.path.join(self.data_dir, "tracklets.h5")
        return tables.open_file(path, 'r')

    def save_tracklets(self, tracklets, app_normalizer, pairs_same, pairs_diff):
        """Saves tracklets to track directory.
        """
        path = os.path.join(self.data_dir, "tracklets.h5")
        h5file = tables.open_file(path, 'w')
        same_grp = h5file.create_group('/', 'pairs_same')
        diff_grp = h5file.create_group('/', 'pairs_diff')
        app_grp = h5file.create_group('/', 'app_fea')
        st_grp = h5file.create_group('/', 'st_fea')
        vid_h, vid_w, _ = self.video_size()
        fps = self.video_frame_rate()
        for seq_len in tracklets:
            gname = 'seq_len_{:04d}'.format(seq_len)
            app = np.array([[
                list(app_normalizer(d)) for d in t
            ] for t in tracklets[seq_len]])
            h5file.create_array(app_grp, gname, app) 
            st = np.array([[[
                float(d['x']) / float(vid_w),
                float(d['y']) / float(vid_h),
                float(d['w']) / float(vid_w),
                float(d['h']) / float(vid_h),
                float(d['frame']) / float(fps)
            ] for d in t] for t in tracklets[seq_len]])
            h5file.create_array(st_grp, gname, st)
            h5file.create_array(same_grp, gname, pairs_same[seq_len])
            h5file.create_array(diff_grp, gname, pairs_diff[seq_len])
        h5file.close()
