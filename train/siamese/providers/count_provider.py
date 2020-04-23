from collections import defaultdict
from random import sample
from multiprocessing import Process
from multiprocessing import Queue
import itertools
from keras.utils import to_categorical
import numpy as np
from utilities import chunks
import time

def _fill_queue(queue, count_data, folds, batch_size):
    # Determine which indices we can use.
    indices = defaultdict(list)
    num_examples = count_data.num_examples()
    for label in num_examples:
        idx_chunks = chunks(range(num_examples[label]), 10)
        for fold in folds:
            indices[label] += idx_chunks[fold]
    print("Number of ignore examples: {}".format(len(indices['ignore'])))
    print("Number of entering examples: {}".format(len(indices['entering'])))

    # Create output vector.
    cats = ['entering', 'ignore'] #'exiting']
    cat_sizes = [int(batch_size / len(cats)) for _ in cats]
    cat_sizes[-1] = batch_size - (len(cats) - 1) * cat_sizes[0]
    out = [[n for _ in range(k)] for n, k in enumerate(cat_sizes)]
    out = list(itertools.chain.from_iterable(out))
    out_cat = to_categorical(out)

    # Start building examples.
    examples = count_data.examples()
    while True:
        samples = [sample(indices[c], n) for c, n in zip(cats, cat_sizes)]
        fea_head = [examples.head[c][s, :, :] for c, s in zip(cats, samples)]
        fea_tail = [examples.tail[c][s, :, :] for c, s in zip(cats, samples)]
        fea_head = np.concatenate(fea_head, axis=0)
        fea_tail = np.concatenate(fea_tail, axis=0)
        """
        # Augment the width and height
        aug_shape = (fea_head.shape[0], fea_head.shape[1], 2)
        fea_head[:, :, 2:4] *= np.random.uniform(0.8, 1.2, aug_shape)
        fea_tail[:, :, 2:4] *= np.random.uniform(0.8, 1.2, aug_shape)
        # Augment the position
        offs_head = fea_head[:, :, 2:4] * np.random.uniform(-0.2, 0.2, aug_shape)
        fea_head[:, :, :2] += offs_head
        offs_tail = fea_tail[:, :, 2:4] * np.random.uniform(-0.2, 0.2, aug_shape)
        fea_tail[:, :, :2] += offs_tail
        """
        queue.put(([fea_head, fea_tail], out_cat))

class CountProvider:
    def __init__(self, count_data, folds, batch_size):
        """Initializes with interface to count data.

        # Arguments
            count_data: Interface to count data directory.
            folds: List of folds (0-9) that can be used.
            batch_size: Batch size.
        """
        self.queue = Queue(10)
        proc = Process(
            target=_fill_queue,
            args=(self.queue, count_data, folds, batch_size))
        proc.daemon = True
        proc.start()

    def generate(self):
        """Yields a batch for training.
        """
        while True:
            yield self.queue.get()
