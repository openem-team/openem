""" Class for finding ruler masks in raw images """
from openem.models import ImageModel
from openem.models import Preprocessor

class RulerMaskFinder(ImageModel):
    """ Class for finding ruler masks from raw images """
    preprocessor=Preprocessor(1.0/128.0,
                              np.array([-1,-1,-1]),
                              True)

    def addImage(image):
        """ Add an image to process in the underlying ImageModel after 
            running preprocessing on it specific to this model. 
 
        image: np.ndarray the underlying image (not pre-processed) to add
               to the model's current batch
        """
        return self._addImage(image, preprocessor)

    
    
