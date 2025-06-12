from abc import ABC, abstractmethod
from PIL import Image

class ObjectCaptioner(ABC):

    @abstractmethod
    def generate_objects(self, image: Image.Image) -> list[str]:
        raise NotImplementedError
        