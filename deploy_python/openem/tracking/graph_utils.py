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
        # TODO: Make this parameterized from strategy
        tid0 = np.argwhere(np.logical_and(diffs > 0, diffs <= max_frame_diff)) #diffs <= max_diffs))
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
