from collections import defaultdict
import numpy as np
from datetime import datetime
from graph import Graph
from graph import FloatVec
from graph import LongVec
from graph import LongPair
from graph import PairVec
from graph import constrainedGreedyAdditiveEdgeContraction
import progressbar
import math

def constant_length(list_seq, timesteps, num_fea, start=False):
    """ Pads sequence with zeros.
        Inputs:
        list_seq -- List of feature vectors.
        timesteps -- Constant number of timesteps to be enforced.
        num_fea -- Number of features in each feature vector.
        start -- If true, uses beginning of sequence for sequences longer
        than desired length, otherwise uses the end of sequence.
    """
    seq = np.array(list_seq)
    if seq.size == 0:
        return np.zeros((timesteps, num_fea))
    vsize, hsize = seq.shape
    assert hsize == num_fea
    if vsize == timesteps:
        return seq
    elif vsize > timesteps:
        if start:
            return seq[:timesteps, :]
        else:
            return seq[-timesteps:, :]
    else:
        return np.vstack((np.zeros((timesteps - vsize, num_fea)), seq))

def renumber_track_ids(track_ids):
    uids, index = np.unique(track_ids, return_index=True)
    id_map = {}
    for new_id, uid in enumerate(uids[np.argsort(index)]):
        id_map[uid] = new_id
    new_ids = [id_map[t] for t in track_ids]
    return new_ids

def _make_tracklets(detections, track_ids):
    tracklets = defaultdict(list)
    num_tracklets = np.max(track_ids) + 1
    assert(len(detections) == len(track_ids))
    for d, tid in zip(detections, track_ids):
        tracklets[tid].append(d)
    return list(tracklets.values())

def _find_edge_pairs(tracklets, max_frame_diff):
    pairs = []
    start_frames = np.array([int(t[0]['frame']) for t in tracklets])
    stop_frames = np.array([int(t[-1]['frame']) for t in tracklets])
    lengths = np.array([len(t) for t in tracklets])
    length_based_max_diffs = np.clip(2 * lengths, 0, max_frame_diff)
    for tid1, start in enumerate(start_frames):
        diffs = start - stop_frames
        max_diffs = np.clip(length_based_max_diffs, 0, length_based_max_diffs[tid1])
        tid0 = np.argwhere(np.logical_and(diffs > 0, diffs <= max_diffs))
        pairs += [(t[0], tid1) for t in tid0]
    return pairs

def _find_constraints(tracklets):
    constraints = []
    frame_ranges = np.array([
        [int(t[0]['frame']), int(t[-1]['frame'])] for t in tracklets
    ])
    for tid0, (start0, stop0) in enumerate(frame_ranges):
        in_range = np.logical_and(frame_ranges >= start0, frame_ranges <= stop0)
        in_range = np.logical_or(in_range[:, 0], in_range[:, 1])
        tid1 = np.argwhere(in_range)
        constraints += [(tid0, t[0]) for t in tid1 if t[0] != tid0]
    return constraints

class HybridWeights:
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

def compute(tracklets,pairs):
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
                    float(tracklets[t0][k]['frame']) / float(fps)
                ] for k in range(-num_dets, 0)])
                d1_st = np.array([[
                    float(tracklets[t1][k]['x']) / float(vid_w),
                    float(tracklets[t1][k]['y']) / float(vid_h),
                    float(tracklets[t1][k]['w']) / float(vid_w),
                    float(tracklets[t1][k]['h']) / float(vid_h),
                    float(tracklets[t1][k]['frame']) / float(fps)
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

def _tracklets_to_ids(tracklets, track_ids):
    detections = []
    det_ids = []
    assert(len(tracklets) == len(track_ids))
    for t, tid in zip(tracklets, track_ids):
        detections += t
        det_ids += [tid for _ in range(len(t))]
    det_ids = renumber_track_ids(det_ids)
    return (detections, det_ids)

def join_tracklets(
        detections,
        track_ids,
        max_frame_diff,
        weight_strategy
    ):
    print(f"{datetime.now()}: Renumbering track IDs...")
    new_ids = renumber_track_ids(track_ids)
    print(f"{datetime.now()}: Creating tracklets...")
    tracklets = _make_tracklets(detections, new_ids)
    print(f"{datetime.now()}: Finding edges...")
    pairs = _find_edge_pairs(tracklets, max_frame_diff)
    print(f"{datetime.now()}: Finding constraints...")
    constraints = _find_constraints(tracklets)
    print(f"{datetime.now()}: Computing edge weights...")
    weights = weight_strategy.compute(tracklets,
                                      pairs)
    print(f"{datetime.now()}: Constructing graph...")
    graph = Graph()
    graph.insertVertices(len(set(new_ids)))
    for p0, p1 in pairs:
        graph.insertEdge(int(p0), int(p1))
    weights_vec = FloatVec()
    for w in weights:
        weights_vec.append(w)
    constraints_vec = PairVec()
    for c0, c1 in constraints:
        constraints_vec.append(LongPair(int(c0), int(c1)))
    arg = LongVec()
    print(f"{datetime.now()}: Solving graph...")
    constrainedGreedyAdditiveEdgeContraction(graph, weights_vec, constraints_vec, arg)
    print(f"{datetime.now()}: Aggregating edge cut status...")
    is_cut = [arg[int(p0)] != arg[int(p1)] for p0, p1 in pairs]
    print(f"{datetime.now()}: Converting back to detection list...")
    new_dets, new_ids = _tracklets_to_ids(tracklets, arg)
    print(f"{datetime.now()}: Iteration complete!")
    return (new_dets, new_ids, pairs, weights, is_cut, constraints)

