from multiprocessing import Process
from multiprocessing import Queue
from random import randint
from random import randrange
from random import choices
import numpy as np
from utilities import constant_length

def _random_example_from_file(track_data, queue, timesteps):
    h5file = track_data.tracklets()
    seq_lens = [
        [s._v_name for s in h5file.root.pairs_same],
        [s._v_name for s in h5file.root.pairs_diff]
    ]
    num_seq_lens = [len(seq_len) for seq_len in seq_lens]
    while True:
        cls = randint(0, 1)
        if num_seq_lens[cls] == 0:
            continue
        seq = randrange(num_seq_lens[cls])
        seq_len = seq_lens[cls][seq]
        if cls == 0:
            num_pairs = h5file.root.pairs_same[seq_len].shape[0]
            if num_pairs == 0:
                continue
            pair = randrange(num_pairs)
            t0, t1 = h5file.root.pairs_same[seq_len][pair]
        else:
            num_pairs = h5file.root.pairs_diff[seq_len].shape[0]
            if num_pairs == 0:
                continue
            pair = randrange(num_pairs)
            t0, t1 = h5file.root.pairs_diff[seq_len][pair]
        a0 = h5file.root.app_fea[seq_len][t0]
        a1 = h5file.root.app_fea[seq_len][t1]
        s0 = h5file.root.st_fea[seq_len][t0]
        s1 = h5file.root.st_fea[seq_len][t1]
        start_frame = np.min(s0[:, -1])
        s0[:, -1] = s0[:, -1] - start_frame
        s1[:, -1] = s1[:, -1] - start_frame
        a0 = constant_length(a0, timesteps, a0.shape[-1])
        a1 = constant_length(a1, timesteps, a1.shape[-1])
        s0 = constant_length(s0, timesteps, s0.shape[-1])
        s1 = constant_length(s1, timesteps, s1.shape[-1])
        queue.put((a0, s0, a1, s1, float(cls)))

class TrackletProvider:
    
    def __init__(self, track_data_list, batch_size, timesteps):
        """Loads tracklet pair indices and sets variables for sampling them.
        """
        self.track_data_list = track_data_list
        num_examples = [t.num_tracklet_pairs() for t in track_data_list]
        total_examples = sum(num_examples)
        self.prob = [float(n) / float(total_examples) for n in num_examples]
        self.queues = [Queue(128) for _ in track_data_list]
        self.procs = [
            Process(
                target=_random_example_from_file,
                args=(track_data, queue, timesteps)
            )
            for track_data, queue in zip(track_data_list, self.queues)
        ]
        for p in self.procs:
            p.daemon = True
            p.start()
        self.batch_size = batch_size

    def generate(self):
        """Builds a batch.
        """
        while True:
            queues = choices(self.queues, self.prob, k=self.batch_size)
            batch = [queue.get() for queue in queues]
            app0_batch, st0_batch, app1_batch, st1_batch, out_batch = zip(*batch)
            app0_batch = np.stack(app0_batch)
            st0_batch = np.stack(st0_batch)
            app1_batch = np.stack(app1_batch)
            st1_batch = np.stack(st1_batch)
            out_batch = np.array(out_batch)
            yield ([app0_batch, st0_batch, app1_batch, st1_batch], out_batch)
