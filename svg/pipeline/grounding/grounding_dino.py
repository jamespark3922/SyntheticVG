from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

from ..utils import create_region_annotations
from . import GroundingModel

class GroundingDinoSAM(GroundingModel):
    """ Grounding DINO model with SAM mask generator """
    
    def __init__(self, pretrained_model_name_or_path: str, sam_model, device, box_threshold=0.15, text_threshold=0.15):
        from segment_anything import SamPredictor
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).to(device)
        self.device = device
        self.mask_generator = SamPredictor(sam_model)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
    
    def generate_regions(self, image: Image.Image, objects: list[str]) -> list[dict]:
        text = ". ".join(objects)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get grounded regions
        result: dict = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        boxes = result['boxes'].cpu().numpy()
        if len(boxes) == 0:
            return []
        
        # Generate masks for the boxes
        image_rgb = np.asarray(image)
        masks = self.generate_masks(image_rgb, boxes)

        regions = create_region_annotations(masks, boxes)

        # TODO: Add object labels to regions
        return regions
    
    def generate_masks(self, image_rgb: np.ndarray, boxes) -> np.ndarray:
        """
        Generate masks for the given boxes

        Args:
            image_rgb (np.ndarray): input image [H x W x 3]
            boxes: N x 4 array of boxes 

        Returns:
            np.ndarray: binary segmentation masks [N x H x W]
        """
        self.mask_generator.set_image(image_rgb)
        input_boxes = torch.tensor(boxes, device=self.mask_generator.device)
        transformed_boxes = self.mask_generator.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])
        masks, _, _ = self.mask_generator.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return masks.squeeze(1).cpu().numpy()