""" Module for performing classification of a detection """
import numpy as np
import cv2
import csv

from openem.models import ImageModel
from openem.models import Preprocessor
from openem.image import crop

from collections import namedtuple
Classification=namedtuple('Classification', ['species', 'cover', 'frame', 'video_id'], defaults=[None,None])

class IO:
    def from_csv(filepath_like):
        classifications=[]
        with open(filepath_like, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # Skip header
            last_idx = -1
            for row in reader:
                col_cnt=len(row)
                species_start=2
                species_end=col_cnt-3
                species=row[species_start:species_end]
                cover=row[species_end:]
                for idx,el in enumerate(species):
                    species[idx] = float(el)
                for idx,el in enumerate(cover):
                    cover[idx] = float(el)

                item=Classification(frame=row[0],
                                    video_id=row[1],
                                    species=species,
                                    cover=cover)
                frame_num = int(float(row[0]))
                if last_idx == frame_num:
                    classifications[last_idx].append(item)
                else:
                    # Add Empties
                    for _ in range(frame_num-1-last_idx):
                        classifications.append([])
                    classifications.append([item])
                    last_idx = frame_num

        return classifications
class Classifier(ImageModel):
    preprocessor=Preprocessor(1.0/127.5,
                              np.array([-1,-1,-1]),
                              True)

    def __init__(self, model_path, gpu_fraction=1.0):
        """ Initialize an image model object
        model_path : str or path-like object
                     Path to the frozen protobuf of the tensorflow graph
        gpu_fraction : float
                       Fraction of GPU allowed to be used by this object.
        """
        super(Classifier, self).__init__(model_path, gpu_fraction,
                                       'data:0',
                                       ['cat_species_1:0',
                                        'cat_cover_1:0'])
    def addImage(self, image):
        """ Add an image to process in the underlying ImageModel after
            running preprocessing on it specific to this model.

        image: np.ndarray the underlying image (not pre-processed) to add
               to the model's current batch
        """
        return self._addImage(image, self.preprocessor)

    def process(self):
        tensors = super(Classifier, self).process()
        if tensors is None:
            return tensors

        # These tensors are potentially batched by image
        species = tensors[0]
        cover = tensors[1]

        results_by_image=[]
        for image_idx, image_species in enumerate(species):
            image_cover = cover[image_idx]
            classification = Classification(species=image_species,
                                            cover=image_cover)
            results_by_image.append(classification)
        return results_by_image
        

    
