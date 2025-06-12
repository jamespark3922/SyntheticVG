import logging
from enum import Enum
from typing import Literal, Union
import torch
import re
from PIL import Image 
from torchvision.ops import masks_to_boxes

from transformers import AutoTokenizer
from svg.mm_utils import tokenizer_image_token
from svg.constants import (
    IMAGE_TOKEN_INDEX, BOX_START, BOX_END,
    BOX_START_QWEN2, BOX_END_QWEN2
)
from svg.mm_utils import (
    tokenizer_image_token,
    process_anyres_image, process_highres_image, process_highres_image_crop_split
)
from svg.conversation import conv_templates, get_stop_str
from svg.model.language_model.robin_qwen import RobinQwenForCausalLM
from svg.model.multimodal_encoder.clip_encoder import  get_image_processor
from svg.utils import disable_torch_init
from svg.train.train import DataArguments, preprocess_multimodal

from .utils import annToMask

from functools import partial
import os
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Constants and Prefix Strings
# ------------------------------------------------------------------------------
IMAGE_STRING = '<image>\nThis provides an overview of the picture.\n'
SINGLE_REGION_STRING = 'There is region1 <mask><pos> in the image.\n'
REGION_PREFIX = 'There are {} regions in the image: <region>.\n'
# Questions
CAPTION = 'Please explain in detail the scene depicted in the picture.'
SG_QUESTION = 'Generate an extremely detailed scene graph for given regions.'
SG_E2E_QUESTION = 'Generate scene graph with proposed regions in bbox coordinates.'
RELATION_DESCRIPTION_QUESTION = 'Generate a description for: {region}.'
RELATION_QUESTION = 'Generate list of relationships for: {region}.'
SUMMARY = 'Generate a summary for the image.'

# ------------------------------------------------------------------------------
# Demo Type Enumeration
# ------------------------------------------------------------------------------
class DemoType(Enum):
    CAPTION = 'caption'
    DESCRIPTION_RELATION = 'description_relation'
    DESCRIPTION = 'description'
    RELATION = 'relation'
    HOLISTIC_SG = 'holistic_sg'

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def get_masks_from_regions(regions) -> np.ndarray:
    """Extracts numpy masks from a list of region dictionaries."""
    return np.array([annToMask(region['segmentation']) for region in regions])

def xywh2xyxy(xywh):
    x,y,w,h = xywh
    return [x, y, x+w, y+h]

def denormalize_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    '''
    Denormalizes bbox (x1,y1,x2,y2) from 0 to 1000 to absolute pixel values.
    '''
    try:
        x1,y1,x2,y2 = bbox
    except ValueError:
        logger.error('Failed to parse bbox:', bbox)
        return None
    x1 = x1 / 1000 * width
    y1 = y1 / 1000 * height
    x2 = x2 / 1000 * width
    y2 = y2 / 1000 * height

    return [int(x1), int(y1), int(x2), int(y2)]

def mask_to_box(mask: np.ndarray | torch.Tensor) -> list[float]:
    """ Converts a mask (2D/3D) into a bounding box [x1, y1, x2, y2]. """
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)
    if mask.dim() == 2:
        mask.unsqueeze_(0)
    
    assert mask.dim() == 3, f"mask dim should be 3, got {mask.dim()}"

    if mask.sum() == 0:
        logger.warning("Warning: mask sum is 0")
        return [0, 0, 0, 0]
    bbox = masks_to_boxes(mask)[0].numpy()
    return bbox

def normalize_bbox(bbox: list[float | int], height: int, width: int) -> list[float]:
    """ Normalizes bbox coordinates from absolute values to 0-1 range. """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
    return [x1, y1, x2, y2] 

def square_pad_bbox(bbox: np.ndarray) -> np.ndarray:
    """ 
    Pads a bbox so that it is square.
    Example:
        bbox = np.array([0, 0, 100, 200])
        bbox = square_pad_bbox(bbox)
        print(bbox)
        # Output: [0, 50, 100, 150]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w > h:
        diff = w - h
        y1 -= diff // 2
        y2 += diff - diff // 2
    elif h > w:
        diff = h - w
        x1 -= diff // 2
        x2 += diff - diff // 2
    return np.array([x1, y1, x2, y2])

def get_whole_image_mask(height, width) -> torch.Tensor:
    """
    Get a mask for the whole image. 
    Used when masks are None.
    """
    mask = np.ones((height, width), dtype=np.bool_)
    mask = mask.astype(np.float32)
    mask = torch.tensor(mask).unsqueeze(0)
    return mask

def parse_scene_graph(scene_graph_output: str) -> dict[str, list]:
    """
    Args:
        scene_graph_output (str): Raw scene graph output from the language model.
        Format example:
        '''
        Objects: 
        region1: <description>
        region2: <description>
        ...
        Relations:
        region1: region2 held by, region3 next to
        '''

    Returns:
        dict: Parsed scene graph with objects and relations
        {
            'objects': list[str],  # List of object descriptions
            'relations': list[tuple[int, int, str]],  # [subject_id, object_id, relation_name]
        }
    """
    # Extract objects section using regex
    objects_pattern = r"Objects:(.*?)(?=Relations:|$)"
    objects_match = re.search(objects_pattern, scene_graph_output, re.DOTALL)
    if not objects_match:
        logger.warning('Failed to parse objects from scene graph output: {}'.format(scene_graph_output))
        return None
    
    # Parse individual object descriptions
    objects_raw = objects_match.group(1)
    object_pattern = r"region\d+:\s*(.*?)(?=\s*(?:region\d+:|$))"
    object_outputs = re.findall(object_pattern, objects_raw, re.DOTALL)
    object_outputs: list[str] = [re.sub(r'<.*?>', '', obj).strip() for obj in object_outputs]

    def get_region_id(s: str) -> int:
        m = re.search(r'region(\d+)', s)
        return int(m.group(1)) - 1 if m else -1

    # Parse relations
    relation_triplets = []
    relations_match = re.search(r"Relations:(.*)", scene_graph_output, re.DOTALL)
    if relations_match:
        relations_raw = relations_match.group(1)
        for line in relations_raw.strip().split('\n'):
            if ':' not in line:
                continue
            source_part, targets = line.split(':', 1)
            src_id = get_region_id(source_part)
            if src_id == -1 or src_id >= len(object_outputs):
                continue
            for item in targets.split(','):
                item = item.strip()
                parts = item.split(' ', 1)
                if len(parts) == 2:
                    tgt, rel = parts
                    tgt_id = get_region_id(tgt) 
                    if tgt_id == -1 or tgt_id >= len(object_outputs):
                        continue
                    relation_triplets.append((src_id, tgt_id, rel))
                
    return {
        'objects': object_outputs,
        'relations': relation_triplets,
    }
        
# ------------------------------------------------------------------------------
# Main Pipeline Class
# ------------------------------------------------------------------------------
class RobinPipeline:
    def __init__(self, model_path: str, device='cuda', **kwargs):

        disable_torch_init()
        model_path = os.path.expanduser(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.unk_token
        special_tokens = ['<mask>', '<pos>', '<image>']
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_processor = get_image_processor(img_size=512)
        self.model = RobinQwenForCausalLM.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.bfloat16,
                                                ).to(device)
    
        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device=device)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.device = device
    
        conv = 'qwen_2'
        self.is_qwen2 = True
        self.conv = conv_templates[conv]
        self.stop_str = get_stop_str(self.conv)

        # Set data args
        data_args = DataArguments()
        data_args.mm_use_im_start_end = False
        data_args.is_multimodal = True
        data_args.image_aspect_ratio = self.model.config.image_aspect_ratio
        self.data_args = data_args

        # Region parameters.
        self.region_mode = kwargs.get('region_mode', 'segmentation')
        self.use_bbox_text = kwargs.get('use_bbox_text', True)
        self.max_regions = kwargs.get('max_regions', 99)
    
    def load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """Loads an image from a file path or PIL object and converts it to RGB PIL."""
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        return Image.open(image).convert('RGB')
    
    def _get_masks(self, masks: np.ndarray) -> torch.Tensor:
        """
        Ensures that the masks are a torch.Tensor on the proper device with 3 dimensions.
        """
        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks)
        if not isinstance(masks, torch.Tensor):
            raise ValueError('Invalid type for masks:', type(masks))
        
        return masks.to(self.model.device)
    
    def _get_new_conv(self):
        return self.conv.copy()
    
    def _get_region_string(self, n: int, bbox_texts: list[str]=None) -> str:
        """
        Constructs a prefix string based on the number of regions and optional bbox texts.
        """
        if bbox_texts is not None:
            assert len(bbox_texts) == n
        
        ref_string = ''
        for i in range(n):
            if not bbox_texts:
                ref_string = ref_string +  f'region{i+1} <mask><pos>, '
            else:
                ref_string = ref_string +  f'region{i+1} <mask><pos> {bbox_texts[i]}, '
        ref_string = ref_string[:-2] # remove the last comma

        ref_prefix = REGION_PREFIX.format(n)
        region_string = ref_prefix.replace('<region>', ref_string)
        region_string = region_string

        return region_string
    
    def _norm_bbox_to_text(self, norm_bbox: list[float | int], scale=1000) -> str:
        """ 
        Returns a text representation of a normalized bbox.
        Assumes bbox is normalized to 0-1000 in [x1,y1,x2,y2] format.
        """
        bbox = np.array(norm_bbox)*scale
        bbox = [int(i) for i in bbox]

        if self.is_qwen2:
            # bbox_text = <|box_start|>(x1,y1),(x2,y2)<|box_end|>
            bbox_text = f'{BOX_START_QWEN2}({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]}){BOX_END_QWEN2}'
        else:
            # bbox_text = [x1,y1,x2,y2]
            bbox_text = f'[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]'
        return bbox_text
    
    def bbox_to_text(self, bboxes: np.ndarray, height: int, width: int) -> str:
        """
        Converts one or more bboxes into a text representation (0-1000 normalized, xyxy).
        Args:
            bboxes: xyxy raw bbox, not normalized. # ([N, 4] or [4])
            height: image height
            width: image width
        Returns:
            str: bbox text used in prompt
        """
        bboxes = np.array(bboxes)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, 0)
        assert bboxes.ndim == 2, f"bbox dim should be 2, got {bboxes.ndim}" # [N, 4]
        
        # Normalize each bbox and convert to text
        bbox_texts = []
        for bbox in bboxes:
            norm_bbox = normalize_bbox(bbox, height, width)
            square_resize = self.data_args.image_aspect_ratio == 'square'
            if square_resize:
                norm_bbox = square_pad_bbox(norm_bbox, height, width)
            bbox_text = self._norm_bbox_to_text(norm_bbox)
            bbox_texts.append(bbox_text)
        
        if self.is_qwen2: 
            bboxes_text = ''.join(bbox_texts)
        else:
            bboxes_text = ','.join(bbox_texts)
            bboxes_text = f'{BOX_START}[{bboxes_text}]{BOX_END}'
        return bboxes_text
    
    def mask_to_bbox_text(self, masks: np.ndarray | torch.Tensor) -> str:
        """
        Converts one or more masks into a text representation of bounding boxes.
        Args:
            masks: [N, H, W] or [H, W]
        Returns:
            str: bbox text used in prompt
        """
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                masks = np.expand_dims(masks, 0)
            assert masks.ndim == 3, f"mask dim should be 3, got {masks.ndim}"
        elif isinstance(masks, torch.Tensor):
            if masks.dim() == 2:
                masks = masks.unsqueeze(0)
            assert masks.dim() == 3, f"mask dim should be 3, got {masks.dim()}"

        bboxes = np.array([mask_to_box(mask) for mask in masks]) # absolute xyxy
        h, w = masks.shape[1:3]
        return self.bbox_to_text(bboxes, h, w)
    
    def get_region_prefix(self, inputs: list[dict] | np.ndarray) -> str:
        """
        Constructs prefix string describing number of regions, mask token positions, 
            and bbox texts for each region if `use_bbox_text` has been set in init.

        Args:
            inputs (list[dict] | np.ndarray): The regions or masks for the image.

        Returns:
            str: A prefix string to use.
        """

        n = len(inputs)
        assert n > 0, 'No regions provided.'
        if self.use_bbox_text:
            if isinstance(inputs, np.ndarray):
                bbox_texts = [self.mask_to_bbox_text(mask) for mask in inputs]
            else:
                masks = get_masks_from_regions(inputs)
                if len(inputs) and 'bbox' in inputs[0]:
                    h,w = masks[0].shape
                    bboxes =  [xywh2xyxy(region['bbox']) for region in inputs]
                    bbox_texts = [self.bbox_to_text(bbox, h, w) for bbox in bboxes]
                else:
                    bbox_texts = [self.mask_to_bbox_text(mask) for mask in masks]
            return self._get_region_string(n, bbox_texts)
        else:
            return self._get_region_string(n)
    
    def generate_scene_graph(
        self, image: Image.Image, inputs: list[dict] | np.ndarray, **kwargs
    ) -> tuple[dict, str]:
        """
        Args:
            image (str): PIL image object.
            inputs (list[dict] | np.ndarray): input regions or masks. ([N] dict or [N, H, W] array)
            kwargs: Additional keyword arguments for generation.
        Returns:
            dict: The generated scene graph containing objects and relations.
                ex: {
                    'objects': List of object descriptions
                    'relations': List of relation triplets [subject_id, object_id, relation_name]
                }
                
            str: The generated output.
        """

        prompt = self.get_region_prefix(inputs)
        prompt = prompt + ' ' + SG_QUESTION
        masks: np.ndarray = get_masks_from_regions(inputs)
        masks: torch.Tensor = self._get_masks(masks) # [N, H, W]
        sg_text: str = self.generate(image, masks, prompt, **kwargs)
        sg: dict = self.parse_scene_graph(sg_text)

        return sg, sg_text
    
    def generate_scene_graph_e2e(
        self, image: str | Image.Image, **kwargs
    ) -> tuple[list, dict, str]:
        """
        Generate a scene graph with proposed regions in bbox coordinates.
        Args:
            image (str): The path to the image, or PIL image object.
            kwargs: Additional keyword arguments for generation.
        Returns:
            boxes: list of [x1,y1,x2,y2] bounding boxes in absolute coordinates.
            sg: dict containing objects and relations.
            sg_text: The generated output.
        """ 

        def get_proposed_boxes(region_outputs, height, width) -> list[list[int]]:
            """
            Extracts and processes bounding box proposals from the given region outputs.
            Args:
                region_outputs (str): The string containing the region output data.
                height (int): The height of the image.
                width (int): The width of the image.
            Returns:
                list: A list of [x1,y1,x2,y2] bounding boxes in absolute coordinates.
            """

            if self.is_qwen2:
                bbox_pattern = r'\(\d+\s*,\s*\d+\)\s*,\s*\(\d+\s*,\s*\d+\)' # get text with (x1, y1), (x2, y2) pattern
                bbox_strings = re.findall(bbox_pattern, region_outputs)
            else:
                bbox_pattern = r'\[\[(.*?)\]\]'  # get text within double square brackets
                bbox_strings = re.findall(bbox_pattern, region_outputs)
            
            bbox_regions = [self.get_bbox_from_text(bbox_string) for bbox_string in bbox_strings]
            # Unnormalize the bounding box regions
            boxes = []
            for bbox in bbox_regions:
                bbox[0] = int(bbox[0]/1000 * width)
                bbox[1] = int(bbox[1]/1000 * height)
                bbox[2] = int(bbox[2]/1000 * width)
                bbox[3] = int(bbox[3]/1000 * height)
                boxes.append(bbox)
            return boxes

        width, height = image.size
        prompt = SG_E2E_QUESTION
        sg_text: str = self.generate(image, masks=None, prompt=prompt, **kwargs)
        sg: dict = self.parse_scene_graph(sg_text)
        boxes = get_proposed_boxes(sg_text, height, width)

        return boxes, sg, sg_text
        

    def generate(self, image: str | Image.Image, masks: torch.Tensor, prompt: Union[str, list[str]], 
                temperature=1.0, top_p=1.0, max_new_tokens=1024, **kwargs
    ) -> str:
        """
        Generate a text output based on an image, mask, and prompt.

        Args:
            image (str): The path to the image, or PIL image object.
            masks (torch.Tensor): The masks for the image. [N, H, W]. If None, a mask for the whole image is generated.
            prompt (Union[str, List[str]]): The prompt for generating the text output. A list should contain alternating sources and outputs.
            temperature (float, optional): The temperature for controlling the randomness of the output. Defaults to 1.0.
            top_p (float, optional): The top-p value for controlling the diversity of the output. Defaults to 1.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 1024.

        Returns:
            str: The generated text output.
        """

        if isinstance(prompt, list):
            assert len(prompt) % 2 == 1, 'Prompt should be a list of alternating sources and outputs'
        else:
            prompt = [prompt]
        init_inputs: dict = self._get_init_inputs(
            image,
            prompt=prompt[0],
            masks=masks,
        )
        image = init_inputs['image']
        masks = init_inputs['masks']

        conv = self._get_new_conv()
        qs = init_inputs['sources'][0][0]['value'] # Initial prompt with image tokens
        conv_prompt = [qs] + prompt[1:] +  [None]
        for idx, p in enumerate(conv_prompt):
            role = conv.roles[idx % 2]  # Alternate between user and model roles
            conv.append_message(role, p)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        outputs: str = self._get_outputs(image, input_ids, masks, self.stop_str, 
                                        temperature, top_p, max_new_tokens=max_new_tokens, **kwargs)

        return outputs
    
    # for gradio demo
    def demo(self, image: str | Image.Image, masks: np.ndarray, 
             type: Literal['description_relation', 'description', 'relation', 'holistic_sg'], 
             mask_idx: int = None, 
             temperature: float = 0.2, top_p: float = 1.0, max_tokens: int = 1024) -> str:
        """
        Generates output based on the specified type.

        Args:
            image (str | Image.Image): The path to the image or a PIL image object.
            masks (np.ndarray): The masks for the image. Shape: [N, H, W].
            type (Literal['description_relation', 'description', 'relation', 'holistic_sg']):
                The type of output to generate.
            mask_idx (int, optional): The index of the mask to generate the output for.
                Required for 'description_relation', 'description', and 'relation' types.
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
            top_p (float, optional): Top-p value for nucleus sampling. Defaults to 1.0.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 1024.

        Returns:
            str: The generated output.
        """
        # Validate and convert demo type.
        try:
            demo_type = DemoType(type)
        except ValueError:
            raise ValueError(f"Invalid demo type: {type}")

        if masks is not None:
            region_string = self.get_region_prefix(masks)
            masks = self._get_masks(masks)
        else:
            assert demo_type == DemoType.CAPTION, "Masks must be provided for all demo types except 'caption'"

        # Ensure mask_idx is provided when required.
        if demo_type in {DemoType.DESCRIPTION_RELATION, DemoType.DESCRIPTION, DemoType.RELATION} and mask_idx is None:
            raise ValueError("mask_idx must be provided for 'description_relation', 'description', and 'relation' types")

        # Define functions for each demo type.
        def demo_description_relation():
            desc_question = RELATION_DESCRIPTION_QUESTION.format(region=f'region{mask_idx+1}')
            prompt_desc = region_string + ' ' + desc_question
            description = self.generate(
                image,
                masks,
                prompt_desc,
                temperature=temperature, top_p=top_p, max_new_tokens=max_tokens
            )
            relation_description = self.generate_relation(
                image,
                masks,
                mask_idx,
                object_output=description
            )
            return f"{description}\n{relation_description}"

        def demo_description():
            question = RELATION_DESCRIPTION_QUESTION.format(region=f'region{mask_idx+1}')
            prompt = region_string + ' ' + question
            return self.generate(
                image,
                masks,
                prompt,
                temperature=temperature, top_p=top_p, max_new_tokens=max_tokens
            )

        def demo_relation():
            question = RELATION_QUESTION.format(region=f'region{mask_idx+1}')
            prompt = region_string + ' ' + question
            return self.generate(
                image,
                masks,
                prompt,
                temperature=temperature, top_p=top_p, max_new_tokens=max_tokens
            )

        def demo_holistic_sg():
            prompt = region_string + ' ' + SG_QUESTION
            return self.generate(
                image,
                masks,
                prompt,
                temperature=temperature, top_p=top_p, max_new_tokens=max_tokens
            )
        
        def demo_caption():
            prompt = CAPTION
            return self.generate(
                image,
                masks,
                prompt,
                temperature=temperature, top_p=top_p, max_new_tokens=max_tokens
            )

        # Mapping from demo types to corresponding functions.
        demo_functions = {
            DemoType.DESCRIPTION_RELATION: demo_description_relation,
            DemoType.DESCRIPTION: demo_description,
            DemoType.RELATION: demo_relation,
            DemoType.HOLISTIC_SG: demo_holistic_sg,
            DemoType.CAPTION: demo_caption
        }

        return demo_functions[demo_type]()
    
    def generate_relation(self, 
        image: Image.Image, 
        masks: torch.Tensor, 
        region_id: int,
        object_output: str=None, 
        bbox_texts=None,
        temperature=0.2, 
        top_p=1.0, 
        max_new_tokens=1024
    ) -> str:
        """ Describes relations between objects in given regions for specified region_id. """

        num_objects = len(masks)
        assert region_id < num_objects

        begin_string = self._get_region_string(n=num_objects, bbox_texts=bbox_texts) 
        subj_region: str = 'region' + str(region_id+1)

        # Create prompt for relation generation
        relation_question = RELATION_QUESTION.format(subj_region)
        if object_output is not None: # Use object description as context
            prompt: str = begin_string + ' ' + RELATION_DESCRIPTION_QUESTION
            prompt = prompt.format(subj_region)
            init_inputs: dict = self.get_init_inputs(image,
                                        self.image_processor,
                                        masks=masks,
                                        prompt=prompt,
                                        )
            image = init_inputs['image']
            masks = init_inputs['masks'].to(self.device)
            conv = self._get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], object_output)
            conv.append_message(conv.roles[0], relation_question)
            conv.append_message(conv.roles[1], None)
        else: # Generate relations without object description context
            prompt: str = begin_string + ' ' + relation_question
            prompt = prompt.format(subj_region)
            init_inputs: dict = self.get_init_inputs(image,
                                    self.image_processor,
                                    masks=masks,
                                    prompt=prompt,
                                    )
            image = init_inputs['image']
            masks = init_inputs['masks'].to(self.device)
            conv = self._get_new_conv()
            qs = init_inputs['sources'][0][0]['value']
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # logging.debug(f"Detailed Relation Prompt: {prompt}")

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        relation_output: str = self.get_outputs(image, input_ids, masks, self.stop_str, 
                                        temperature, top_p, max_new_tokens=max_new_tokens)
        relation_output = f"{subj_region}: {relation_output}"

        # logging.debug(f"Relation Output: {relation_output}")

        return relation_output
    
    def parse_scene_graph(self, scene_graph_output: str) -> dict[str, list]:
        return parse_scene_graph(scene_graph_output)
    
    def get_bbox_from_text(self, text: str) -> list[int]:
        """ 
        Extracts a single bbox from text.
        
        Example:
            text = "<box>(716,0),(1000,283)</box>" # qwen2 format
            bbox = get_bbox_from_text(text)
            print(bbox)
            # Output: [716, 0, 1000, 283]
        """ 
        # Define the regular expression to match the pattern (x1,y1),(x2,y2)
        pattern = r'\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)'
        
        # Find the match in the text
        match = re.search(pattern, text)
        
        if not match:
            logger.error('Failed to parse bbox:', text)
            return []
        
        try:
            # Extract the coordinates and convert them to integers
            bbox = [int(match.group(i)) for i in range(1, 5)]
        except ValueError:
            logger.error('Failed to parse bbox:', text)
            return []
        
        return bbox
    
    def _get_init_inputs(self,
                        image: Union[str, Image.Image],
                        prompt: str,
                        masks: torch.Tensor = None,
            ):
        """
        Get the initial inputs for evaluation.

        Args:
            image (Union[str, Image.Image]): The path to the image or the image object itself.
            prompt (str): The prompt for the evaluation.
            masks (torch.Tensor): The masks for the image.

        Returns:
            dict: A dictionary containing the processed inputs for evaluation.
        """

        def get_image_token_len(image):
            patch_size: int = 16 # Default: 16
            image_token_len = 0
            for im in image:
                image_token_len += (im.shape[0] // patch_size) * (im.shape[1] // patch_size)
            return image_token_len
           
        image: Image.Image = self.load_image(image)
        w,h = image.size

        image_aspect_ratio = self.model.config.image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt", do_center_crop=False)["pixel_values"][0].unsqueeze(0)
        else:
            image = self.image_processor.preprocess(image, return_tensors="pt", do_center_crop=False)["pixel_values"][0]
            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False)

        image_token_len = get_image_token_len(image)

        begin_str = IMAGE_STRING
        if masks is None:
            begin_str = begin_str + SINGLE_REGION_STRING
            masks = get_whole_image_mask(h, w)
        
        prompt = begin_str + prompt
        masks = masks.to(self.model.device)

        sources = dict()
        sources['conversations'] = []
        sources['conversations'].append({'from': 'human', 'value': prompt})
        
        sources = preprocess_multimodal([sources['conversations']], self.data_args, image_token_len)

        data_dict = {}
        data_dict['sources'] = sources
        data_dict['image'] = image
        data_dict['masks'] = masks
        return data_dict
    
    def _get_outputs(
        self, 
        image: torch.Tensor, 
        input_ids: torch.Tensor, 
        masks: torch.Tensor, 
        # past_key_values=None,
        stop_str: str = None, 
        temperature=1.0, 
        top_p=1.0, 
        max_new_tokens=512, 
        skip_special_tokens=True,
        do_sample=True,
        **kwargs
    ) -> str:
        """
        Generate outputs based on the given input.

        Args:
            image (torch.Tensor): The input image.
            input_ids (torch.Tensor): The input token IDs.
            masks (torch.Tensor): The segmentation masks.
            stop_str (str): The stop string to indicate the end of generation.
            temperature (float, optional): The temperature for sampling. Defaults to 1.0.
            top_p (float, optional): The top-p value for nucleus sampling. Defaults to 1.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            do_sample (bool, optional): Whether to use sampling or greedy decoding. Defaults to True.

        Returns:
            str: The generated outputs.
        """
        if stop_str is None:
            stop_str = get_stop_str(self.conv)
            
        self.model.model.tokenizer = self.tokenizer
        
        masks = [masks.half().to(self.device) if masks is not None else None]
        if self.conv.version in ['qwen', 'mistral']:
            output = self.model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().to(self.device),
                    masks=masks,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    use_cache=True,
                    num_beams=1,
                    **kwargs
                )
            outputs = self.tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)[0]
        
        else:
            with torch.inference_mode():
                
                self.model.orig_forward = self.model.forward
                self.model.forward = partial(
                    self.model.orig_forward,
                    img_metas=[None],
                    masks=masks
                )
                output = self.model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    use_cache=True,
                    num_beams=1,
                    **kwargs
                )

                self.model.forward = self.model.orig_forward
            output_ids = output

            # Parse output response
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                logger.warning(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                skip_special_tokens=skip_special_tokens)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs: str = outputs.strip()

        return outputs
    
    def _get_relation_triplets(self, relations: list[str]) -> list[tuple[int, int, str]]:
        def get_region_id(region_str: str) -> int:
            """
            Extracts 1-indexed region ID from a string and convert to 0-indexed int.
            """
            match = re.search(r'region(\d+)', region_str)
            if match:
                return int(match.group(1)) - 1
            return -1
        
        def get_triplet(relation_str: str):
            """
            Extracts the triplet (source, relation, target) from a relation string.

            Update [4/3/24]:
                Generated output has the format: 'region<id> relation, ...'. 
                This is useful for getting relation calibration score for given object_id.
            """
            parts = relation_str.split(':')
            source_id: int = get_region_id(parts[0])

            # Find all occurrences of patterns like "region{id} relation"
            if 'region' in parts[1]:
                if ',' in parts[1]:
                    # Handling multiple relations for the same source region
                    relations = parts[1].split(',')
                    for relation in relations:
                        relation = relation.strip()
                        try:
                            target_str, relation_type = relation.split(' ', 1)
                            target_id = get_region_id(target_str)
                            yield (source_id, target_id, relation_type)
                        except ValueError as e:
                            # print('Failed to parse relation: {}'.format(relation))
                            continue
                else:
                    try:
                        target_str, relation_type = parts[1].strip().split(' ', 1)
                        target_id = get_region_id(target_str)
                        yield (source_id, target_id, relation_type)
                    except ValueError as e:
                        logger.warning('Failed to parse relation: {}'.format(relation_str))
                        pass
            
        triplets = []
        for rel in relations:
            if rel.count(':') == 1:
                for triplet in get_triplet(rel):
                    triplets.append(triplet)
        return triplets

    


    