from abc import ABC, abstractmethod
from PIL import Image 

class GroundingModel:
    """ Grounding model interface """
    
    @abstractmethod
    def generate_regions(self, image: Image.Image, objects: list[str]) -> list[dict]:
        """_summary_

        Args:
            image (Image.Image): _description_
            objects (list[str]): _description_

        Returns:
            list[dict]: List of region annotations, each containing:
            - segmentation: RLE encoded mask
            - bbox: [x, y, width, height] format
            - area: number of pixels in mask
        """
        pass