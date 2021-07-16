""" Various methods to compute weights """

import numpy as np
import cv2
import math
import progressbar

class HybridWeights:
    """ Method uses CNN / RNN based on number of detections present """
    def __init__(self,
                 det_comparator,
                 track_comparator,
                 app_normalizer,
                 vid_dims,
                 fps,
                 single_frame_bias,
                 batch_size):
        self.det_comparator = det_comparator
        self.track_comparator = track_comparator
        self.app_normalizer = app_normalizer
        self.vid_dims = vid_dims
        self.fps = fps
        self.single_frame_bias = single_frame_bias
        self.batch_size = batch_size

    def compute(self, tracklets,pairs):
        """ CNN/RNN hybrid strategy """
        timesteps = 24
        weights = [0.0 for _ in pairs]
        bar=progressbar.ProgressBar(max_value=len(pairs), prefix="weights")
        detection_w_idx=[]
        detection_w_d0=[]
        detection_w_d1=[]

        detection_y_idx=[]
        detection_y_d0_app=[]
        detection_y_d0_st=[]
        detection_y_d1_app=[]
        detection_y_d1_st=[]

        vid_w = self.vid_dims[1]
        vid_h = self.vid_dims[0]

        for i, (t0, t1) in bar(enumerate(pairs)):
            num_dets = min(len(tracklets[t0]), len(tracklets[t1]))
            num_dets = min(num_dets, timesteps)
            if num_dets == 1:
                # Single pair run can use rgb lookup for now
                # TODO: Switch this to appearence extraction to be
                # consistent
                d0 = tracklets[t0][-1]
                d1 = tracklets[t1][0]
                detection_w_idx.append(i)
                detection_w_d0.append(d0)
                detection_w_d1.append(d1)
            else:
                d0_app = np.array([self.app_normalizer(tracklets[t0][k]) for k in range(-num_dets, 0)])
                d1_app = np.array([self.app_normalizer(tracklets[t1][k]) for k in range(0, num_dets)])
                d0_st = np.array([[
                    float(tracklets[t0][k]['x']) / float(vid_w),
                    float(tracklets[t0][k]['y']) / float(vid_h),
                    float(tracklets[t0][k]['w']) / float(vid_w),
                    float(tracklets[t0][k]['h']) / float(vid_h),
                    float(tracklets[t0][k]['frame']) / float(self.fps)
                ] for k in range(-num_dets, 0)])
                d1_st = np.array([[
                    float(tracklets[t1][k]['x']) / float(vid_w),
                    float(tracklets[t1][k]['y']) / float(vid_h),
                    float(tracklets[t1][k]['w']) / float(vid_w),
                    float(tracklets[t1][k]['h']) / float(vid_h),
                    float(tracklets[t1][k]['frame']) / float(self.fps)
                ] for k in range(0, num_dets)])
                assert(d0_st[-1, -1] < d1_st[0, -1])
                min_frame = np.min(d0_st[:, -1])
                d0_st[:, -1] -= min_frame
                d1_st[:, -1] -= min_frame
                d0_app = constant_length(d0_app, timesteps, d0_app.shape[-1])
                d1_app = constant_length(d1_app, timesteps, d1_app.shape[-1])
                d0_st = constant_length(d0_st, timesteps, d0_st.shape[-1])
                d1_st = constant_length(d1_st, timesteps, d1_st.shape[-1])
                detection_y_idx.append(i)
                detection_y_d0_app.append(np.expand_dims(d0_app, axis=0))
                detection_y_d0_st.append(np.expand_dims(d0_st,axis=0))
                detection_y_d1_app.append(np.expand_dims(d1_app,axis=0))
                detection_y_d1_st.append(np.expand_dims(d1_st, axis=0))

        num_batches = math.ceil(len(detection_w_idx)/self.batch_size)
        bar=progressbar.ProgressBar(max_value=num_batches,prefix="batches")
        for bn in bar(range(num_batches)):
            start=bn*self.batch_size
            end=start+self.batch_size
            indices=detection_w_idx[start:end]
            for idx in indices:
                self.det_comparator.addPair(detection_w_d0[idx]['bgr'],
                                       detection_w_d1[idx]['bgr'])
            res = self.det_comparator.process()
            for res_idx, res in enumerate(res):
                # get global idx from result idx
                g_idx=int(indices[res_idx])
                res = res[0]
                orig_res = res
                res += self.single_frame_bias
                # bound between 0 and 1
                res = min(max(res, 0.0), 1.0)
                if res != 0:
                    res = np.log((1.0 - res) / res)
                else:
                    res = 1000000
                res= min(1000000, res)
                res = max(-1000000, res)
                weights[g_idx] = res

        #batch_size=track_comparator.batch
        num_batches = math.ceil(len(detection_y_idx)/self.batch_size)
        bar=progressbar.ProgressBar(max_value=num_batches,prefix="batches")
        for bn in bar(range(num_batches)):
            start=bn*self.batch_size
            end=start+self.batch_size
            indices=detection_y_idx[start:end]
            d0_app=np.vstack(detection_y_d0_app[start:end])
            d0_st=np.vstack(detection_y_d0_st[start:end])
            d1_app=np.vstack(detection_y_d1_app[start:end])
            d1_st=np.vstack(detection_y_d1_st[start:end])
            feed_dict={self.track_comparator.d0_app: d0_app,
                       self.track_comparator.d0_st: d0_st,
                       self.track_comparator.d1_app: d1_app,
                       self.track_comparator.d1_st: d1_st}
            res = self.track_comparator.session.run([self.track_comparator.output],feed_dict=feed_dict)[0][0]

            # Batch logic here would be faster...
            for res_idx, res in enumerate(res):
                # get global idx from result idx
                g_idx=int(indices[res_idx])
                res = np.log((1.0 - res) / res)
                res = min(1000000, res)
                res = max(-1000000, res)
                weights[g_idx] = res

        return weights


class IoUWeights:
    """ Calculate edge weight based on IoU """
    def __init__(self, vid_dims, threshold=0.10):
        self.vid_dims = vid_dims
        self.threshold = threshold
    def _intersection_over_union(self,boxA, boxB):
        """ Computes intersection over union for two bounding boxes.
            Inputs:
            boxA -- First box. Must be a dict containing x, y, width, height.
            boxB -- Second box. Must be a dict containing x, y, width, height.
            Return:
            Intersection over union.
        """
        # normalize to full coordinates
        box_ax = int(boxA["x"]*self.vid_dims[1])
        box_bx = int(boxB["x"]*self.vid_dims[1])
        box_ay = int(boxA["y"]*self.vid_dims[0])
        box_by = int(boxB["y"]*self.vid_dims[0])
        box_aw = int(boxA["width"]*self.vid_dims[1])
        box_bw = int(boxB["width"]*self.vid_dims[1])
        box_ah = int(boxA["height"]*self.vid_dims[0])
        box_bh = int(boxB["height"]*self.vid_dims[0])

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box_ax, box_bx)
        yA = max(box_ay, box_by)
        xB = min(box_ax + box_aw,
                 box_bx + box_bw)
        yB = min(box_ay + box_ah,
                 box_by + box_bh)

        # compute the area of intersection rectangle
        interX = xB - xA + 1
        interY = yB - yA + 1
        if interX < 0 or interY < 0:
            iou = 0.0
        else:
            interArea = float((xB - xA + 1) * (yB - yA + 1))
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = box_aw*box_ah
            boxBArea = box_bw*box_bh

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            if float(boxAArea + boxBArea - interArea) <= 0.0:
                return 0.00
            try:
                iou = interArea / float(boxAArea + boxBArea - interArea)
            except Exception as e:
                print(e)
                print("interArea: {}".format(interArea))
                print("Union: {}".format(float(boxAArea + boxBArea - interArea)))
            # return the intersection over union value
        return iou

    def compute(self, tracklets, pairs):
        weights = [0.0 for _ in pairs]
        for weight_idx, (t0, t1) in enumerate(pairs):
            # Pick the last of the 1st tracklet
            # and the first of the 2nd tracklet
            d0 = tracklets[t0][-1]
            d1 = tracklets[t1][0]
            iou = self._intersection_over_union(d0, d1)
            iou = min(iou,1.0)
            if iou > self.threshold:
                # threshold to 1.0 translates to 0 to 1000000
                weights[weight_idx] = math.pow(1000000,iou)
            else:
                weights[weight_idx] = -1000000
        return weights

def track_vel(track):
    track.sort(key=lambda x:x['frame'])
    track_len = track[-1]['frame'] - track[0]['frame']

    if 'orig_x' in track[0]:
        f_cx = track[0]['orig_x'] + (track[0]['orig_w']/2)
        f_cy = track[0]['orig_y'] + (track[0]['orig_h']/2)
    else:
        f_cx = track[0]['x'] + (track[0]['width']/2)
        f_cy = track[0]['y'] + (track[0]['height']/2)

    if 'orig_x' in track[-1]:
        l_cx = track[-1]['orig_x'] + (track[-1]['orig_w']/2)
        l_cy = track[-1]['orig_y'] + (track[-1]['orig_h']/2)
    else:
        l_cx = track[-1]['x'] + (track[-1]['width']/2)
        l_cy = track[-1]['y'] + (track[-1]['height']/2)

    print(f"{track[0]['frame']}: {f_cx},{f_cy} to {l_cx,l_cy}")
    x_vel = (l_cx-f_cx)  / track_len
    y_vel = (l_cy - f_cy) / track_len
    magnitude=math.sqrt(math.pow(x_vel,2)+math.pow(y_vel,2))
    if magnitude <= 0.00001:
        magnitude = 0
        angle = 0
        x_vel = 0
        y_vel = 0
    else:
        angle = math.atan2(y_vel, x_vel)
        # unfurl radian
        if angle < 0:
            angle = 2*math.pi + angle
    return (angle, magnitude,[x_vel,y_vel])

class IoUMotionWeights(IoUWeights):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def size_similarity(self, track1, track2):
        if 'orig_h' in track1[-1]:
            area_1 = track1[-1]['orig_w'] * track1[-1]['orig_h']
        else:
            area_1 = track1[-1]['width'] * track1[-1]['height']
        if 'orig_w' in track2[0]:
            area_2 = track2[0]['orig_w'] * track2[0]['orig_h']
        else:
            area_2 = track2[0]['width'] * track2[0]['height']
        return 1.0 - (abs(area_1-area_2)/area_2)

    def motion_similarity(self,track1,track2):
        if len(track1) < 4 or len(track2) < 4:
            return 0.0

        track1.sort(key=lambda r:r['frame'])
        track2.sort(key=lambda r:r['frame'])
        frame = 0
        for t in track1:
            if t['frame'] < frame:
                print("Out of order detections")
            else:
                frame = t['frame']

        track1_vel = track_vel(track1)
        track2_vel = track_vel(track2)

        angle_diff = math.atan2(math.sin(track1_vel[0]-track2_vel[0]),math.cos(track1_vel[0]-track2_vel[0]))
        angle_diff /= math.pi * 2
        if angle_diff >= 0.25:
            score = -1.0
        elif angle_diff >= 0.05:
            score = 1.0 - angle_diff
        else:
            score = 1.0
        print(f"{track1_vel}, {track2_vel}: Angle_diff = {angle_diff}. Velocity score = {score}")
        return score
    def compute(self, tracklets, pairs):
        weights = [0.0 for _ in pairs]
        for weight_idx, (t0, t1) in enumerate(pairs):
            # Pick the last of the 1st tracklet
            # and the first of the 2nd tracklet
            tracklets[t0].sort(key=lambda x:x['frame'])
            tracklets[t1].sort(key=lambda x:x['frame'])
            d0 = tracklets[t0][-1]
            d1 = tracklets[t1][0]
            iou = self._intersection_over_union(d0, d1)
            iou = min(iou,1.0)
            if iou > self.threshold:
                motion = self.motion_similarity(tracklets[t0],tracklets[t1])
                size = self.size_similarity(tracklets[t0], tracklets[t1])
                # threshold to 1.0 translates to 0 to 1000000
                if motion >= 0:
                    weights[weight_idx] = math.pow(1000000,(iou*0.25)+(motion*0.50)+(size*0.25))
                else:
                    # If motion is rejected, discard the edge
                    weights[weight_idx] = -1000000
                print(f"m:{motion} i:{iou} s:{size} -> {weights[weight_idx]}")
            else:
                weights[weight_idx] = -1000000
        return weights

class IoUGlobalMotionWeights(IoUWeights):
    """ Does a global motion prediction based on phase correlation of bounding box centers.
    """
    def __init__(self, vid_dims, media_file, **kwargs):
        self._media_file = media_file
        self._shifts = None
        super().__init__(vid_dims, **kwargs)

    def compute(self, tracklets, pairs):
        vid_h, vid_w = self.vid_dims
        # Compute global offsets.
        if self._shifts is None:
            # Compute ROI for global offsets.
            x0 = [det['x'] for track in tracklets for det in track]
            y0 = [det['y'] for track in tracklets for det in track]
            x1 = [det['x'] + det['width'] for track in tracklets for det in track]
            y1 = [det['y'] + det['height'] for track in tracklets for det in track]
            if len(x0) > 2:
                x0 = int(np.percentile(x0, 10, interpolation='nearest') * vid_w)
                y0 = int(np.percentile(y0, 10, interpolation='nearest') * vid_h)
                x1 = int(np.percentile(x1, 90, interpolation='nearest') * vid_w)
                y1 = int(np.percentile(y1, 90, interpolation='nearest') * vid_h)
                compute_shifts = True
                print(f"ROI for phase correlation (x0, y0, x1, y1): {x0}, {y0}, {x1}, {y1}")
            else:
                compute_shifts = False
            # Use phase correlation to compute shifts.
            vid = cv2.VideoCapture(self._media_file)
            prev = None
            self._shifts = []
            while True:
                ok, frame_bgr = vid.read()
                if not ok:
                    break
                height, width, _ = frame_bgr.shape
                if (prev is None) or (not compute_shifts):
                    self._shifts.append((0.0, 0.0))
                else:
                    img0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    img1 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    (dx, dy), _ = cv2.phaseCorrelate(img0[y0:y1, x0:x1], img1[y0:y1, x0:x1])
                    if dx > width / 4.0:
                        dx -= width / 2.0
                    if dy > height / 4.0:
                        dy -= height / 2.0
                    self._shifts.append((dx, dy))
                    print(f"Frame {len(self._shifts)} offsets: DX={dx}, DY={dy}")
                prev = frame_bgr
        # Compute IOU between pairs using global shifts.
        weights = [0.0 for _ in pairs]
        for weight_idx, (t0, t1) in enumerate(pairs):
            # Pick the last of the 1st tracklet
            # and the first of the 2nd tracklet
            d0 = dict(tracklets[t0][-1])
            d1 = tracklets[t1][0]
            for frame in range(d0['frame'], d1['frame']):
                dx, dy = self._shifts[frame + 1]
                d0['x'] += dx / vid_w
                d0['y'] += dy / vid_h
            iou = self._intersection_over_union(d0, d1)
            iou = min(iou, 1.0)
            if iou > self.threshold:
                # threshold to 1.0 translates to 0 to 1000000
                weights[weight_idx] = math.pow(1000000, iou)
            else:
                weights[weight_idx] = -1000000
        return weights

