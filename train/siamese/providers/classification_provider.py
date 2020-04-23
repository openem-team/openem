from collections import defaultdict
from random import sample
from multiprocessing import Process
from multiprocessing import Queue
import itertools
from keras.utils import to_categorical
import numpy as np
from utilities import chunks
import time

def _fill_queue(queue, class_data, folds, batch_size):

    # Get list of species.
    cats = class_data.species_list()

    # Determine which indices we can use.
    indices = defaultdict(list)
    num_examples = class_data.num_examples()
    for label in num_examples:
        idx_chunks = chunks(range(num_examples[label]), 10)
        for fold in folds:
            indices[label] += idx_chunks[fold]
    for species in cats:
        print("Number of {} examples: {}".format(species, len(indices[species])))

    # Create output vector.
    cat_sizes = [int(batch_size / len(cats)) for _ in cats]
    cat_sizes[-1] = batch_size - (len(cats) - 1) * cat_sizes[0]
    out = [[n for _ in range(k)] for n, k in enumerate(cat_sizes)]
    out = list(itertools.chain.from_iterable(out))
    out_cat = to_categorical(out)

    # Start building examples.
    examples = class_data.examples()
    while True:
        samples = [sample(indices[c], n) for c, n in zip(cats, cat_sizes)]
        fea = [examples[c][s, :, :] for c, s in zip(cats, samples)]
        fea = np.concatenate(fea, axis=0)
        queue.put((fea, out_cat))

class ClassificationProvider:
    """ Class for generating batches of balanced classification sequences.
    """
    def __init__(self, class_data, folds, batch_size):
        """ Constructor.
        # Arguments
            class_data: Interface to example sequences.
            folds: List of folds (0-9) that can be used.
            batch_size: Batch size.
        """
        self.queue = Queue(10)
        proc = Process(
            target=_fill_queue,
            args=(self.queue, class_data, folds, batch_size))
        proc.daemon = True
        proc.start()

    def generate(self):
        """Yields a batch for training.
        """
        while True:
            yield self.queue.get()
