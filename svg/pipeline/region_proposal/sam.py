import numpy as np
from typing import Literal
from PIL import Image

def load_sam_model(sam_ckpt, device, vit_type='vit_h'):
    from segment_anything import sam_model_registry
    sam = sam_model_registry[vit_type](checkpoint=sam_ckpt).to(device)
    return sam

class SamMaskGenerator:
    def __init__(self, sam_model, output_mode: Literal["coco_rle", "binary_mask"]="coco_rle"):
        """
        Load different SAM mask generators for different region modes.
        Args:
            sam_model: SAM model to use
            output_mode: output mode of the mask generator. Either 'coco_rle' or 'binary_mask'
                - 'coco_rle': RLE encoded mask in string type
                - 'binary_mask': binary mask in numpy array [H, W]
        """

        # Load SAM mask generators
        from som.task_adapter.sam.tasks.automatic_mask_generator import load_sam_mask_generator
        self.default_mask_model = load_sam_mask_generator('default', sam_model, output_mode="coco_rle")
        self.part_mask_model = load_sam_mask_generator('part', sam_model, output_mode="coco_rle")
        self.whole_mask_model = load_sam_mask_generator('whole', sam_model, output_mode="coco_rle")
    
    def get_mask_generator(self, region_mode: str):
        if region_mode == 'default':
            return self.default_mask_model
        elif region_mode == 'part':
            return self.part_mask_model
        elif region_mode == 'whole':
            return self.whole_mask_model
        else:
            raise ValueError(f"Invalid region mode: {region_mode}. Available region modes: ['default', 'part', 'whole']")
    
    def generate_regions(self, image: Image.Image, region_mode: str = 'default') -> list[dict]:
        """
        Generate regions from an image using the SAM model.
        
        Args:
            image (Image.Image): input image
            region_mode (str, optional): type of sam mask generator to run. Defaults to 'default'.

        Returns:
            list[dict]: list of regions with following keys:
                - segmentation (dict(str, any) or np.ndarray): The mask. If
                    output_mode='binary_mask', is an array of shape HW. Otherwise,
                    is a dictionary containing the RLE.
                - bbox (list(float)): The box around the mask, in XYWH format.
                - area (int): The area in pixels of the mask.
                - predicted_iou (float): The model's own prediction of the mask's
                    quality. This is filtered by the pred_iou_thresh parameter.
                - point_coords (list(list(float))): The point coordinates input
                    to the model to generate this mask.
                - stability_score (float): A measure of the mask's quality. This
                    is filtered on using the stability_score_thresh parameter.
                - crop_box (list(float)): The crop of the image used to generate
                    the mask, given in XYWH format.
        """
        mask_generator = self.get_mask_generator(region_mode)

        image_rgb = np.asarray(image)
        regions = mask_generator.generate(image_rgb)
        return regions