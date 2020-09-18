import glob
import os
import tables

class CountData:
    """Interface to count data.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir.strip()

    def json_files(self):
        """Returns list of json files in the directory.
        """
        return glob.glob(os.path.join(self.data_dir, '*.json'))

    def clear_examples(self):
        """Clears examples.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        if os.path.exists(path):
            os.remove(path)

    def num_examples(self):
        """Returns number of tracklet pairs.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        h5file = tables.open_file(path, 'r')
        num_rows = {
            'entering': h5file.root.head['entering'].nrows,
            'ignore': h5file.root.head['ignore'].nrows,
            'exiting': h5file.root.head['exiting'].nrows
        }
        h5file.close()
        return num_rows

    def examples(self):
        """Returns training examples for 3 categories.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        return tables.open_file(path, 'r').root

    def create_examples(self, timesteps, nfea):
        """Creates file to store spatiotemporal examples.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        h5file = tables.open_file(path, 'w')
        for side in ['head', 'tail']:
            group = h5file.create_group('/', side)
            h5file.create_earray(
                group,
                'entering',
                atom=tables.FloatAtom(),
                shape=(0, timesteps, nfea),
                expectedrows=10000
            )
            h5file.create_earray(
                group,
                'exiting',
                atom=tables.FloatAtom(),
                shape=(0, timesteps, nfea),
                expectedrows=1000
            )
            h5file.create_earray(
                group,
                'ignore',
                atom=tables.FloatAtom(),
                shape=(0, timesteps, nfea),
                expectedrows=10000
            )
        h5file.close()

    def save_example(self, fea_head, fea_tail, out):
        """Saves example to count file.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        h5file = tables.open_file(path, 'a')
        h5file.root.head[out].append(fea_head)
        h5file.root.tail[out].append(fea_tail)
        h5file.close()
