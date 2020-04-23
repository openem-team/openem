import pickle
import random
import numpy as np

class SequenceProvider:
    """ Class for generating a sequence.  Uses masking to enforce constant
        sequence length.
    """
    def __init__(
        self, 
        trk_list, 
        assoc_examples,
        sequence_size,
        batch_size, 
        extractors,
        do_plots=False,
        model_data=None):
        """ Constructor.
            Inputs:
            trk_list -- List of lists of track objects (one list per 
                directory).
            assoc_examples -- List of lists, each containing directory
                index, track id, sequence length, detection, and output.
            sequence_size - Maximum sequence length.
            batch_size - Number of examples to include in batch.
            extractors - OrderedDict containing feature name as key and 
                extractor as value.
            do_plots - True to do plots.
            model_data - Required to do plots.
        """
        ## List of lists of track objects.
        self.trk_list = trk_list
        ## Unique outputs.
        self.unique_outputs = np.unique([ex[4] for ex in assoc_examples])
        ## Number of classes.
        self.num_classes = len(self.unique_outputs)
        ## Num class by num examples by 5 nested list.
        self.assoc_examples = [
            [ex for ex in assoc_examples if ex[4] == out] 
            for out in self.unique_outputs]
        ## Current example index per class.
        self.example_index = [0 for _ in range(self.num_classes)]
        ## Number of examples per class.
        self.num_class_examples = [len(a) for a in self.assoc_examples]
        ## Sequence size.
        self.sequence_size = sequence_size
        ## Batch size.
        self.batch_size = batch_size
        ## Feature extractor.
        self.extractors = extractors
        ## Whether to do plotting.
        self.do_plots = do_plots
        ## Images if needed for plotting.
        if do_plots:
            self.det_imgs = model_data.detection_images()
                
    def __next_example(self, c):
        """ Returns next example.  Once the end is reached, the examples
            are shuffled.
            Inputs:
            c -- The desired class index.
        """
        if self.example_index[c] == 0:
            print("Started new cycle through class with label " +
                "{}, permuting...".format(self.unique_outputs[c]))
            random.shuffle(self.assoc_examples[c])
        ex = self.assoc_examples[c][self.example_index[c]]
        self.example_index[c] += 1
        num_ex = self.num_class_examples[c]
        self.example_index[c] = self.example_index[c] % num_ex
        return ex

    def __get_feature(self, det, det_seq, feature):
        """ Retrieves the appropriate feature from a detection.
            Inputs:
            det - A detection.
            det_seq - Sequence of all preceding detections in the track.
            feature - Feature to extract.
            Returns:
            Feature vector.
        """
        if feature == "appearance":
            return self.extractors["appearance"](det)
        elif feature == "motion":
            last = None
            if len(det_seq) > 0:
                last = det_seq[-1]
            return self.extractors["motion"](det, last)
        else:
            raise ValueError(
                "Invalid feature selection {}!".format(feature))

    def __build_sequence(self, example, feature):
        """ Builds a sequence for the given track ID and association index.
            Inputs:
            example - List containing directory index, track id, number of 
                detections in track when associated, associated detection, 
                and output.
            feature - Feature for which sequence should be built.
            Returns:
            Feature sequence, association, output, detection sequence, 
                detection being associated.
        """
        dir_idx, trk_id, seq_len, det_assoc, output = example
        for track in self.trk_list[dir_idx]:
            if track.id == trk_id:
                break
        sequence = []
        det_seq = []
        for det in track.detections[:seq_len]:
            fea = self.__get_feature(det, det_seq, feature)
            sequence.append(fea)
            det_seq.append(det)
        assoc = self.__get_feature(det_assoc, det_seq, feature)
        sequence = np.array(sequence)
        return (sequence, assoc, output, det_seq, det_assoc)
         
    def __plot(self, seq, assoc, output, det_seq, det_assoc):
        """ Plots a sequence.
            Inputs:
            seq - Sequence of feature vectors.
            assoc - List of possible associations to the sequence.
            output - 0 for same, 1 for different.
            det_seq - Sequence of detections.
            det_assoc - Sequence of potential associations.
        """
        if not self.do_plots:
            return
        import matplotlib.pyplot as plt
        num_assoc = len(assoc)
        num_seq = len(seq)
        f = plt.figure()
        f.suptitle("Associations")
        for idx, (a, da, out) in enumerate(zip(assoc, det_assoc, output)):
            plot_idx = idx + 1
            ax = f.add_subplot(2, num_assoc, plot_idx)
            ax.plot(a)
            ax.set_title("Label = {}".format(out))
            ax = f.add_subplot(2, num_assoc, num_assoc + plot_idx)
            ax.imshow(self.det_imgs[da["index"]])
        f = plt.figure()
        f.suptitle("Sequence")
        for idx, (s, d) in enumerate(zip(seq, det_seq)):
            plot_idx = idx + 1
            ax = f.add_subplot(2, num_seq, plot_idx)
            ax.plot(s)
            ax = f.add_subplot(2, num_seq, num_seq + plot_idx)
            ax.imshow(self.det_imgs[d["index"]])
        plt.show()

    def __constant_sequence_length(self, sequence):
        """ Enforces a constant sequence length using mask values to
            fill empty parts of the sequence.
            Inputs:
            sequence - The sequence with variable size.
        """
        num_timesteps, num_features = sequence.shape
        if num_timesteps == self.sequence_size:
            return sequence
        elif num_timesteps > self.sequence_size:
            return sequence[-self.sequence_size:]
        else:
            num_missing = self.sequence_size - num_timesteps
            maskable = np.zeros((num_missing, num_features))
            return np.vstack((maskable, sequence))

    def __call__(self):
        """ Generator.
        """
        batch_seq = [[] for _ in self.extractors]
        batch_assoc = [[] for _ in self.extractors]
        batch_output = []
        c = 0
        while True:
            if len(batch_output) == self.batch_size:
                batch_seq = [np.array(b) for b in batch_seq]
                batch_assoc = [np.array(b) for b in batch_assoc]
                batch_output = np.array(batch_output)
                yield (batch_seq + batch_assoc, batch_output)
                batch_seq = [[] for _ in self.extractors]
                batch_assoc = [[] for _ in self.extractors]
                batch_output = []
            ex = self.__next_example(c)
            for i, feature in enumerate(self.extractors):
                seq, assoc, output, det_seq, det_assoc = \
                    self.__build_sequence(ex, feature)
                self.__plot(seq, assoc, output, det_seq, det_assoc)
                seq = self.__constant_sequence_length(seq)
                batch_seq[i].append(seq)
                batch_assoc[i].append(assoc)
            batch_output.append(output)
            c += 1
            c = c % self.num_classes

