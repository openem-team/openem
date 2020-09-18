import numpy as np

class DataProvider:
    """ Class for filling a queue with data. Handles multiple inputs
        and outputs.
    """
    def __init__(
        self,
        data,
        get_img,
        mean_img,
        augs,
        batch_size,
        batch_queue,
        do_plots=False,
        vgg_scale=True):
        """ Constructor.
            Inputs:
            data -- Numpy archive consisting of the following key/values:
                index: num examples by num inputs array of indices into imgs.
                outputs: num examples by num outputs array of outputs.
            get_img -- Function with the following signature:
                get_img(index) -> img
                Where index is a single integer index and img is the image.
            mean_img -- The mean image.
            augs -- List of functions used to augment data.  One augmentation
                is randomly chosen for each input example.
            batch_size -- Number of examples to include in batch.
            batch_queue -- Queue for storing batches.
            do_plots -- True to plot random input example for each batch.
        """
        mean_img = np.expand_dims(mean_img, axis=0)
        ## Image accessor.
        self.get_img = get_img
        ## Mean image.
        self.mean_img = mean_img
        ## Whether to scale images for VGG input
        self.vgg_scale = vgg_scale
        ## Number of examples.
        self.num_examples = data["index"].shape[0]
        assert(data["outputs"].shape[0] == self.num_examples)
        ## List of unique outputs.
        self.unique_outputs = np.unique(data["outputs"],axis=0).tolist()
        ## Number of classes.
        self.num_classes = len(self.unique_outputs)
        ## Number of inputs.
        self.num_inputs = data["index"].shape[1]
        ## Number of outputs.
        self.num_outputs = data["outputs"].shape[1]
        ## Number of augmentations.
        self.num_augs = len(augs)
        ## Augmentations.
        self.augs = augs
        ## Index into indices for each class.
        self.current_index = [0 for _ in self.unique_outputs]
        ## Indices associated with each class.  List of arrays with
        ## size num class examples by num inputs.
        self.index = []
        for c in self.unique_outputs:
            same_class = np.squeeze(np.equal(data["outputs"], c))
            if len(same_class.shape) > 1:
                same_class = np.squeeze(np.equal(
                        np.sum(same_class,axis=1),
                        len(self.unique_outputs)))
            self.index.append(data["index"][same_class, :])
        ## Number examples of each class.
        self.num_class_examples = [len(a) for a in self.index]
        ## Number of examples per class per batch.
        self.num_examples_per_batch = [
            int(np.floor(batch_size / self.num_classes))
            for _ in self.unique_outputs]
        self.num_examples_per_batch[-1] = batch_size - np.sum(
            self.num_examples_per_batch[:-1])
        ## Number of cycles performed through each class.
        self.cycle = [-1 for _ in self.unique_outputs]
        ## Predefined batch outputs.
        self.batch_outputs = None
        for n, o in zip(self.num_examples_per_batch, self.unique_outputs):
            new_arr = np.tile(o, (n,1))
            if self.batch_outputs is None:
                self.batch_outputs = new_arr
            else:
                self.batch_outputs = np.concatenate((
                    self.batch_outputs, new_arr), axis=0)
        ## Batch queue.
        self.batch_queue = batch_queue
        ## Whether to do plots.
        self.do_plots = do_plots

    def __shuffle(self):
        """ Checks if any of the classes has started a new cycle.  If so,
            the indices are shuffled.
        """
        for c in range(self.num_classes):
            idx = np.float(self.current_index[c])
            num = np.float(self.num_class_examples[c])
            this_cycle = int(np.floor(idx / num))
            if this_cycle > self.cycle[c]:
                print("Started new cycle through class " +
                    "{}, permuting...".format(c))
                np.random.shuffle(self.index[c])
                self.cycle[c] = this_cycle

    def __get_batch(self):
        """ Gets batch of images.  Returns a tensor with size:
            num classes x num inputs x num_imgs x image_dims
        """
        img_batch = []
        for c in range(self.num_classes):
            idx = np.arange(
                self.current_index[c],
                self.current_index[c] + self.num_examples_per_batch[c])
            idx = np.mod(idx, self.num_class_examples[c])
            # batch_idx will have dimensions:
            # num class examples per batch x num inputs
            batch_idx = self.index[c][idx]
            class_batch = [
                [self.get_img(batch_idx[n][i]) for i in range(self.num_inputs)]
                for n in range(self.num_examples_per_batch[c])]
            img_batch.append(class_batch)
        img_batch = np.array(img_batch)
        img_batch = np.swapaxes(img_batch, 1, 2)
        return img_batch

    def __augment(self, batch):
        """ Augments a batch.  We apply the same transform to each input
            but randomly vary the transform between class and example.
        """
        for c in range(self.num_classes):
            aug_funcs = [self.augs[np.random.randint(0, self.num_augs)]
                for _ in range(self.num_examples_per_batch[c])]
            for i in range(self.num_inputs):
                for n in range(self.num_examples_per_batch[c]):
                    batch[c][i][n] = aug_funcs[n](batch[c][i][n])
        return batch

    def __plot(self, batch):
        """ Plots all pairs in batch.
        """
        if self.do_plots:
            import matplotlib.pyplot as plt
            for c in range(self.num_classes):
                fig = plt.figure()
                k = 1
                for ind in range(self.num_examples_per_batch[c]):
                    for i in range(self.num_inputs):
                        ax = fig.add_subplot(
                            self.num_examples_per_batch[c],
                            self.num_inputs,
                            k)
                        ax.imshow(batch[c][i][ind])
                        fig.suptitle("Class {} inputs".format(c, i))
                        k += 1
                plt.show()

    def __to_queue(self, batch):
        """ Shapes the batch and pushes it onto the queue.
        """
        inp = []
        for i in range(self.num_inputs):
            imgs = None
            for c in range(self.num_classes):
                if imgs is None:
                    imgs = batch[c][i]
                else:
                    imgs = np.concatenate((imgs, batch[c][i]), axis=0)
            imgs = imgs - np.repeat(self.mean_img, imgs.shape[0], axis=0)
            if self.vgg_scale:
                imgs = np.divide(imgs, 127.5)
            inp.append(imgs)
        self.batch_queue.put((inp, self.batch_outputs))

    def __next_batch(self):
        """ Increments indices to go to next batch.
        """
        for c in range(self.num_classes):
            self.current_index[c] += self.num_examples_per_batch[c]

    def start(self):
        """ Starts pushing data onto queue.
        """
        while True:
            self.__shuffle()
            batch = self.__get_batch()
            batch = self.__augment(batch)
            self.__plot(batch)
            self.__to_queue(batch)
            self.__next_batch()
