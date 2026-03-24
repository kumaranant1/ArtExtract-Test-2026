import numpy as np
from PIL import ImageOps

class SquarePad:
    """
    Pads a PIL image with pure black borders to make it a perfect square.
    """
    def __call__(self, image):
        w, h = image.size
        maxi = np.max([w, h])
        
        return ImageOps.pad(image, (maxi, maxi), color=0)