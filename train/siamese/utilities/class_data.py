import glob
import os
import tables

class ClassData:
    """Interface to classification data.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir.strip()

    def json_files(self):
        """Returns list of json files in the directory.
        """
        return glob.glob(os.path.join(self.data_dir, '*.json'))

    def clear_examples(self):
        """ Deletes existing examples.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        if os.path.exists(path):
            os.remove(path)

    def species_list(self):
        """Returns species for which features were extracted.
        """
        return [node._v_name for node in self.examples()]

    def num_examples(self):
        """Returns number of tracklet pairs.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        h5file = tables.open_file(path, 'r')
        num_rows = {}
        for species in self.species_list():
            num_rows[species] = h5file.root[species].nrows
        h5file.close()
        return num_rows

    def examples(self):
        """Returns training examples for each species.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        return tables.open_file(path, 'r').root

    def create_examples(self, timesteps, nfea, species_names):
        """Creates file to store classification examples.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        h5file = tables.open_file(path, 'w')
        for species in species_names:
            h5file.create_earray(
                '/',
                species,
                atom=tables.FloatAtom(),
                shape=(0, timesteps, nfea),
                expectedrows=100000
            )
        h5file.close()

    def save_example(self, fea, out):
        """Saves example to count file.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        h5file = tables.open_file(path, 'a')
        h5file.root[out].append(fea)
        h5file.close()

    def normalize_examples(self, normalizer):
        """Normalizes stored features using given normalizer.
        """
        path = os.path.join(self.data_dir, "examples.h5")
        h5file = tables.open_file(path, 'r+')
        for node in h5file.root:
            for i in range(node.shape[0]):
                node[i] = normalizer(node[i])
