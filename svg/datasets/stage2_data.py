
import copy
import json
import logging
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes

from svg.constants import (
    BOX_END,
    BOX_END_QWEN2,
    BOX_START,
    BOX_START_QWEN2,
    PATCH_SIZE,
    POINT_END,
    POINT_START,
)
from svg.file_utils import load_image_from_gcs, open_gcs_or_local
from svg.mm_utils import (
    process_anyres_image,
    process_highres_image,
    process_highres_image_crop_split,
)
from svg.train.train import preprocess, preprocess_multimodal

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

def xywh2xyxy(xywh):
    x,y,w,h = xywh
    return [x, y, x+w, y+h]

def get_whole_image_mask(height, width) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.bool_)
    return mask.astype(np.float32)

def sample_data(data: list, target_size: int):
    """
    Sample data to reach a target size:
    - If data_size > len(data): Repeat the entire dataset n times, then sample randomly for the remainder
    - If data_size <= len(data): Just randomly sample data_size items
    
    Args:
        data (list): The original data (list of dictionaries)
        data_size (int): Target dataset size
    
    Returns:
        list: Sampled data as a list of dictionaries
    """

    if target_size <= 0:
        return []
    
    n = target_size // len(data)
    sample_size = target_size % len(data)

    # Copy the data n times if n > 0
    sample_data = []
    if n > 0:
        sample_data += data * n
    if sample_size > 0:
        sample_data += random.sample(data, target_size)
        
    return sample_data

class CustomCOCO(COCO):
    def __init__(self, annotation_file=None):
        """
        Extend Microsoft COCO helper class to load from gcs file.
        Supports loading attribute labels (e.g. PACO, VG)

        Args:
            annotation_file (str): location of annotation file
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.atts,self.imgs = dict(),dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()

            # Support loading gcs annotation
            with open_gcs_or_local(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert isinstance(dataset, dict), 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
    
    def createIndex(self):
        """ Additionally create index for attributes"""
        super().createIndex()

        atts = {}
        if 'attributes' in self.dataset:
            for att in self.dataset['attributes']:
                atts[att['id']] = att
        self.atts = atts
                
        
    def loadAtts(self, ids=[]):
        """
        Load atts with the specified ids.
        :param ids (int array)       : integer ids specifying atts
        :return: atts (object array) : loaded att
        """

        def _isArrayLike(obj):
            return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

        if _isArrayLike(ids):
            return [self.atts[id] for id in ids]
        elif type(ids) == int:
            return [self.atts[ids]]
            
class CustomDataset(Dataset):
    begin_str = "<image>\nThis provides an overview of the picture.\n"
    region_str = "There is region1 <mask><pos> for the entire image.\n"

    def __init__(self,
                 tokenizer=None,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=20,
                 use_bbox_text=False,
                 sample: int | float = None
                 ):
        """ 
        Custom Dataset used for training.
        
        Args:
            - sample: int | float: if int, number of samples to take, if float, fraction of samples to take
        """
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.img_prefix = img_prefix
        self.max_gt_per_img = max_gt_per_img
        self.use_bbox_text = use_bbox_text

        self.is_qwen2 = 'qwen2' in self.tokenizer.name_or_path.lower()            
        data_infos = self.load_annotations(ann_file)

        # Get data according to sample ratio
        if sample is not None:
            target_size = int(sample * len(data_infos)) if isinstance(sample, float) else sample
            data_infos = self.sample_data(data_infos, target_size)
        self.data_infos = data_infos

        # Set up logging if not already set up
        logging.info(f"{self.__class__.__name__} total len: {len(self.data_infos)} sample: {sample}")

        # Debug before load
        data = self.__getitem__(0, debug=True)
        cur_input_ids = data['input_ids']
        mask_idx = torch.nonzero(cur_input_ids==tokenizer.convert_tokens_to_ids(['<mask>'])[0])
        mask_len = len(data['masks']) if data['masks'] is not None else 0
        assert len(mask_idx) == mask_len, "mask num not equal to mask feats"

    def __len__(self):
        return len(self.data_infos)
    
    @staticmethod
    def sample_data(data: list, target_size: int):
        return sample_data(data, target_size)
    
    @staticmethod
    def get_whole_image_mask(height, width) -> torch.Tensor:
        mask = np.ones((height, width), dtype=np.bool_)
        mask = mask.astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)
        return mask
    
    def load_file(self, ann_file) -> list:
        # json or jsonl file
        if ann_file.endswith('.jsonl'):
            data = []
            with open_gcs_or_local(ann_file) as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            with open_gcs_or_local(ann_file) as f:
                return json.load(f)
    
    @staticmethod
    def bboxToMask(box, height, width):
        """
        Creates a boolean mask with the specified bounding box area filled as True.

        Args:
            box (list): [x1, y1, x2, y2] coordinates for the bounding box.
            height (int): Height of the output mask.
            width (int): Width of the output mask.

        Returns:
            np.ndarray: Boolean mask with shape (height, width) with the box area filled as True.
        """
        mask = np.zeros((height, width), dtype=np.bool_)

        # Ensure the coordinates are within the image bounds
        x1, y1, x2, y2 = np.array(box, dtype=int)
        x2, y2 = [min(x2, width), min(y2, height)]
        
        # Fill the box area with True values
        mask[y1:y2, x1:x2] = True
        
        return mask.astype(np.float32)

    @staticmethod
    def bboxToMaskWithBorder(box, height, width, border_thickness=3, internal_mask: np.ndarray=None):
        """
        Creates a mask with a border around the bounding box and optionally fills the internal area.

        Args:
            box (list): [x1, y1, x2, y2] coordinates for the bounding box.
            height (int): Height of the output mask.
            width (int): Width of the output mask.
            border_thickness (int, optional): Thickness of the border to draw around the bounding box. Defaults to 3.
            internal_mask (np.ndarray, optional): Annotation for the internal mask; if None, only the border is drawn. Defaults to None.

        Returns:
            np.ndarray: A numpy array of shape (height, width) with the bounding box border and internal mask.
        """
        mask = np.zeros((height, width), dtype=np.bool_)
        x1, y1, x2, y2 = np.array(box, dtype=int)

        # Ensure the coordinates are within the image bounds
        x2, y2 = [min(x2, width - 1), min(y2, height - 1)]
        x1, y1 = [max(x1, 0), max(y1, 0)]
        
        # Draw the border

        # Top border
        mask[max(y1-border_thickness, 0):y1+border_thickness, max(x1-border_thickness, 0):min(x2+border_thickness, width)] = True
        # Left border
        mask[max(y1-border_thickness, 0):min(y2+border_thickness, height), max(x1-border_thickness, 0):x1+border_thickness] = True
        # Bottom border
        mask[y2-border_thickness:min(y2+border_thickness, height), x1:min(x2+border_thickness, width)] = True
        # Right border
        mask[max(y1-border_thickness, 0):min(y2+border_thickness, height), x2-border_thickness:min(x2+border_thickness, width)] = True
        x2, y2 = [min(x2, width - 1), min(y2, height - 1)]
        x1, y1 = [max(x1, 0), max(y1, 0)]
        
        # Draw the internal mask if provided
        if internal_mask is not None:
            internal_mask = np.logical_and(internal_mask, np.logical_not(mask))
            mask = np.logical_or(mask, internal_mask)
        
        return mask.astype(np.float32)
    
    @staticmethod
    def annToMask(mask_ann, h, w) -> np.ndarray:
        if isinstance(mask_ann, np.ndarray):
            return mask_ann

        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask
    
    @staticmethod
    def mask_to_box(mask: np.ndarray | torch.Tensor) -> list[float]:
        """ Converts segmentation mask to bbox. """
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask)
        if mask.dim() == 2:
            mask.unsqueeze_(0)
        
        assert mask.dim() == 3, f"mask dim should be 3, got {mask.dim()}"

        if mask.sum() == 0:
            print("mask sum is 0")
            return [0, 0, 0, 0]
        bbox = masks_to_boxes(mask)[0].numpy()
        return bbox
    
    def normalize_point(self, point: list[float | int], height: int, width: int) -> list[float]:
        """ Converts point from 0 to 1 to absolute values. """
        x, y = point
        x, y = x/width, y/height
        return [x, y]
    
    def textify_point(self, point: list[float | int], scale=1000) -> str:
        """ Returns scaled point text in format: [x,y]"""
        point = np.array(point)*scale
        point = [int(i) for i in point]
        point_text = f'[{point[0]},{point[1]}]'
        return point_text
    
    def normalize_bbox(self, bbox: list[float | int], height: int, width: int, 
                       clip_bbox=True) -> list[float]:
        """ Converts bbox from 0 to 1 to absolute values. """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
        try:
            assert x1 <= x2 and y1 <= y2, f"Invalid bbox: {bbox}"
            assert 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1, f"Invalid bbox: {bbox}"
        except AssertionError:
            if clip_bbox:
                logging.debug(f"Invalid bbox: {bbox} height: {height} width: {width}")
                x1,y1,x2,y2 = np.clip(np.array([x1,y1,x2,y2]), 0, 1).tolist()
                logging.debug(f"Clipped bbox: {x1,y1,x2,y2}")
            else:
                logging.error(f"Invalid bbox: {bbox} height: {height} width: {width}")
                raise AssertionError(f"Invalid bbox: {bbox}")
        return [x1, y1, x2, y2]
     
    def textify_bbox(self, bbox: list[float | int], scale=1000) -> str:
        """ Returns scaled bbox text in format: [x1,y1,x2,y2]"""
        bbox = np.array(bbox)*scale
        bbox = [int(i) for i in bbox]
        if self.is_qwen2:
            # <|box_start|>(x1,y1),(x2,y2)<|box_end|>
            bbox_text = f'{BOX_START_QWEN2}({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]}){BOX_END_QWEN2}'
        else:
            # [x1,y1,x2,y2]
            bbox_text = f'[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]'
        return bbox_text 
    
    def square_pad_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """ Pads bbox to make it square """
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
    
    def point_to_text(self, points: np.ndarray, height: int, width: int) -> str:
        """
        Returns absolute point xy to text in format: [x,y]
        - point: xy unnormalized point
        - height: image height
        - width: image width
        """
        points = np.array(points)
        if points.ndim == 1:
            points = np.expand_dims(points, 0)
        assert points.ndim == 2, f"point dim should be 2, got {points.ndim}"

        point_texts = []
        for point in points:
            norm_point = self.normalize_point(point, height, width)
            point_text = self.textify_point(norm_point)
            point_texts.append(point_text)
        point_text_list = ','.join(point_texts)
        point_text = f'{POINT_START}[{point_text_list}]{POINT_END}'
        return point_text
    
    def bbox_to_text(self, bboxes: list | np.ndarray, height: int, width: int, clip_bbox: bool=True) -> str:
        """
        Convert bounding boxes to text representation for model input.
        This function takes bounding boxes in absolute coordinates (xyxy format),
        normalizes them, optionally makes them square (if square_resize is enabled),
        and converts them to a text representation suitable for the model.
        The output format depends on the tokenizer:
        - For Qwen models: concatenated bbox text strings
        - For other models: formatted with BOX_START and BOX_END markers and comma-separated
        Args:
            bboxes (list | np.ndarray): Bounding boxes in xyxy format (x1, y1, x2, y2),
                                        with unnormalized coordinates
            height (int): Image height in pixels
            width (int): Image width in pixels
            clip_bbox (bool, optional): Whether to clip bounding boxes to image bounds.
        Returns:
            str: Text representation of bounding boxes, appropriately formatted
                 based on the tokenizer type
        """
        bboxes = np.array(bboxes)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, 0)
        assert bboxes.ndim == 2, f"bbox dim should be 2, got {bboxes.ndim}"
        
        bbox_texts = []
        for bbox in bboxes:
            norm_bbox = self.normalize_bbox(bbox, height, width, clip_bbox=clip_bbox)
            square_resize = self.data_args.image_aspect_ratio == 'square'
            if square_resize:
                norm_bbox = self.square_pad_bbox(norm_bbox)
            bbox_text = self.textify_bbox(norm_bbox)
            bbox_texts.append(bbox_text)
        
        is_qwen = 'qwen2' in self.tokenizer.name_or_path.lower()
        if is_qwen: 
            # bbox_texts = ['<|box_start|>'+bbox_text+'<|box_end|>' for bbox_text in bbox_texts]
            bbox_text = ''.join(bbox_texts)
        else:
            bbox_text_list = ','.join(bbox_texts)
            bbox_text = f'{BOX_START}[{bbox_text_list}]{BOX_END}'
        return bbox_text
    
    def mask_to_bbox_text(self, masks: np.ndarray | torch.Tensor) -> str:
        """ Returns 2D mask to bbox text in format: [[x1,y1,x2,y2]]"""
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                masks = np.expand_dims(masks, 0)
            assert masks.ndim == 3, f"mask dim should be 3, got {masks.ndim}"
        elif isinstance(masks, torch.Tensor):
            if masks.dim() == 2:
                masks = masks.unsqueeze(0)
            assert masks.dim() == 3, f"mask dim should be 3, got {masks.dim()}"

        bboxes = np.array([self.mask_to_box(mask) for mask in masks]) # absolute xyxy
        h, w = masks.shape[1:3]
        return self.bbox_to_text(bboxes, h, w)
    
    def load_annotations(self, ann_file) -> list[dict]:

        self.coco = CustomCOCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue

            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    
    def get_data_info(self, idx):
        return self.data_infos[idx]
    
    def get_ann_info(self, idx):

        img_id = self.get_data_info(idx)['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info
    
    def load_image(self, image: str) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if image.startswith('gs://'):
            image = load_image_from_gcs(image)
        else:
            image: Image.Image = Image.open(image)
        return image
    
    def process_image(self, image: Image.Image | str, overwrite_image_aspect_ratio=None):
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        image = self.load_image(image)
        image: Image.Image = image.convert('RGB')
        
        def get_image_token_len(image):
            patch_size: int = PATCH_SIZE # Default: 16
            image_token_len = 0
            for im in image:
                image_token_len += (im.shape[0] // patch_size) * (im.shape[1] // patch_size)
            return image_token_len

        image_size = image.size # original image size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
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

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt", do_center_crop=False)["pixel_values"][0].unsqueeze(0)
        else:
            image = processor.preprocess(image, return_tensors="pt", do_center_crop=False)["pixel_values"][0].unsqueeze(0)
        assert len(image.shape) == 4
        image_token_len = get_image_token_len(image)
        return image, image_size, image_token_len

    def __getitem__(self, idx, debug=False):

        data_item = self.get_data_item(idx)
        data_dict = self.process_text(data_item=data_item, debug=debug)

        return data_dict
    
    def get_data_item(self, idx):
        data_info = self.get_data_info(idx)
        ann_info = self.get_ann_info(idx)

        img_path = os.path.join(self.img_prefix, data_info['filename'])
        image, image_size, image_token_len = self.process_image(img_path)

        gt_masks = []
        gt_bboxes = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)

            bbox = xywh2xyxy(ann['bbox']) # xywh to xyxy
            gt_bboxes.append(bbox)

            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])

        data_item = dict(
            image = image,
            image_token_len=image_token_len,
            gt_bboxes = gt_bboxes,
            gt_masks = gt_masks,
            gt_labels = gt_labels
        )
        return data_item
    
    def process_text(self, data_item, debug=False):
        image = data_item['image']
        image_token_len = data_item['image_token_len']
        ori_labels = data_item['gt_labels']
        ori_masks = np.array(data_item['gt_masks'])
        ori_masks = torch.from_numpy(ori_masks) 

        ori_bboxes = data_item['gt_bboxes']

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        ori_masks = ori_masks[shuffle_ids]
        ori_labels = [ori_labels[i] for i in shuffle_ids]
        ori_bboxes = [ori_bboxes[i] for i in shuffle_ids]
        ori_h, ori_w = ori_masks[0].shape
        assert len(ori_masks) == len(ori_bboxes) == len(ori_labels)

        sources = dict()
        sources['conversations'] = []

        # Add bbox text
        for i in range(len(ori_labels)):
            question = '<region>'
            if self.use_bbox_text:
                bbox_text = self.bbox_to_text(ori_bboxes[i], ori_h, ori_w)
                question = question.replace('<region>', f'<mask><pos> {bbox_text}')
            else:
                question = question.replace('<region>', '<mask><pos>')
            if i == 0:
                question = self.begin_str + question
            answer = ori_labels[i]
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        assert image.shape[-2] == image.shape[-1]
        # a hard code [] for sources
        sources = preprocess_multimodal(
            copy.deepcopy([sources['conversations']]),
            self.data_args,
            image_token_len
        )
        if debug:
            for conv in sources[0]:
                print(conv['from'])
                print(conv['value'])
                print("=")

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True
            )
        
        # get single
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = ori_masks
        return data_dict

class COCODataset(CustomDataset):

    def __init__(self,
                 tokenizer=None,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=20,
                 use_bbox_text=False,
                 ):

        self.begin_str = '<image>\nIn the conversation below, you simply answer the category name based on what you see ' \
                        'in the imagery inside a particular region. I will give you only one region each time.\n' 
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, use_bbox_text)
    
class PartImagenet(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 use_bbox_text=False,
                 ):

        CAT_CLASSES = (
            'Bottle', 'Biped', 'Quadruped', 'Fish', 'Reptile', 'Bicycle', 'Bird', 'Car', 'Boat', 'Snake', 'Aeroplane'
        )

        SUB_CLASSES = (
            'Tier', 'Hand', 'Wing', 'Mouth', 'Tail', 'Side', 'Fin', 'Engine', 'Foot', 'Head', 'Body', 'Sail', 'Seat'
        )

        begin_str = '<image>\nIn the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, use_bbox_text)
    
class PascalPart(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 use_bbox_text=False,
                 ):

        CAT_CLASSES = ('potted plant', 'aeroplane', 'cow', 'cat', 'bus', 'horse', 'car', 
                    'dog', 'bicycle', 'person', 'bird', 'bottle', 'sheep', 'motorbike')

        SUB_CLASSES = ('eye', 'window', 'cap', 'headlight', 'hand', 'mirror', 'arm', 'plant', 
                    'wheel', 'ear', 'pot', 'foot', 'leg', 'nose', 'body', 'horn', 'handlebar', 
                    'neck', 'license plate', 'paw', 'saddle', 'head', 'muzzle', 'tail', 'wing', 
                    'beak', 'hair', 'torso', 'door', 'mouth')

        begin_str = '<image>\n In the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category:subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, use_bbox_text)

class PacoDataset(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 use_bbox_text=False,
                 ):

        CAT_CLASSES = ('ball', 'basket', 'belt', 'bench', 'bicycle', 'blender', 'book', 
                       'bottle', 'bowl', 'box', 'broom', 'bucket', 
                       'calculator', 'can', 'car', 'carton', 'cellular telephone', 
                       'chair', 'clock', 'crate', 'cup', 'dog', 'drill', 'drum', 'earphone', 
                       'fan', 'glass', 'guitar', 'hammer', 'handbag', 'hat', 'helmet', 'jar', 
                       'kettle', 'knife', 'ladder', 'lamp', 'laptop computer', 'microwave oven', 
                       'mirror', 'mouse', 'mug', 'napkin', 'newspaper', 'pan', 'pen', 'pencil',
                       'pillow', 'pipe', 'plastic bag', 'plate', 'pliers', 'remote control', 'scarf', 
                       'scissors', 'screwdriver', 'shoe', 'slipper', 'soap', 'sponge', 'spoon', 'stool', 
                       'sweater', 'table', 'tape', 'telephone', 'television set', 'tissue paper', 'towel', 
                       'trash can', 'tray', 'vase', 'wallet', 'watch', 'wrench')

        SUB_CLASSES = ('antenna', 'apron', 'arm', 'back', 'back cover', 'backstay', 'bar', 'barrel', 
                       'base', 'base panel', 'basket', 'bezel', 'blade', 'body', 'border', 'bottom', 
                       'bowl', 'bracket', 'bridge', 'brush', 'brush cap', 'buckle', 'bulb', 'bumper', 
                       'button', 'cable', 'camera', 'canopy', 'cap', 'capsule', 'case', 'clip', 'closure', 
                       'colied tube', 'control panel', 'cover', 'cuff', 'cup', 'decoration', 'dial', 'door handle', 
                       'down tube', 'drawer', 'drawing', 'ear', 'ear pads', 'embroidery', 'end tip', 'eraser', 'eye', 
                       'eyelet', 'face', 'face shield', 'fan box', 'fender', 'ferrule', 'finger hole', 'fingerboard', 
                       'finial', 'flap', 'food cup', 'foot', 'footrest', 'fork', 'frame', 'fringes', 'gear', 'grille', 
                       'grip', 'hand', 'handle', 'handlebar', 'head', 'head tube', 'headband', 'headlight', 'headstock', 
                       'heel', 'hem', 'hole', 'hood', 'housing', 'inner body', 'inner side', 'inner wall', 'insole', 'jaw', 
                       'joint', 'key', 'keyboard', 'label', 'lace', 'lead', 'left button', 'leg', 'lid', 'light', 'lining', 
                       'logo', 'loop', 'lower bristles', 'lug', 'mirror', 'motor', 'mouth', 'neck', 'neckband', 'nose', 'nozzle',
                       'nozzle stem', 'outer side', 'outsole', 'page', 'pedal', 'pedestal column', 'pediment', 'pickguard', 'pipe', 
                       'pom pom', 'prong', 'pull tab', 'punt', 'push pull cap', 'quarter', 'rail', 'right button', 'rim', 'ring', 'rod', 
                       'roll', 'roof', 'rough surface', 'runningboard', 'saddle', 'screen', 'screw', 'scroll wheel', 'seal ring', 'seat', 
                       'seat stay', 'seat tube', 'shade', 'shade cap', 'shade inner side', 'shaft', 'shank', 'shelf', 'shoulder', 'side', 
                       'side button', 'sign', 'sipper', 'skirt', 'sleeve', 'slider', 'spindle', 'splashboard', 'spout', 'steeringwheel', 
                       'stem', 'step', 'sticker', 'stile', 'strap', 'stretcher', 'string', 'switch', 'swivel', 'table top', 'tail', 'taillight', 
                       'tank', 'tapering top', 'teeth', 'terry bar', 'text', 'throat', 'time display', 'tip', 'toe box', 'tongue', 'top', 'top cap', 
                       'top tube', 'touchpad', 'trunk', 'turnsignal', 'turntable', 'vamp', 'vapour cover', 'visor', 'welt', 'wheel', 'window', 'windowpane', 
                       'windshield', 'wiper', 'wire', 'yoke', 'zip')

        ATTR_CLASSES = ('black', 'blue', 'brown', 'ceramic', 'checkered', 'crochet', 
                        'dark blue', 'dark brown', 'dark green', 'dark grey', 'dark orange', 
                        'dark pink', 'dark purple', 'dark red', 'dark yellow', 'dotted', 'fabric', 
                        'floral', 'glass', 'green', 'grey', 'leather', 'light blue', 'light brown', 
                        'light green', 'light grey', 'light orange', 'light pink', 'light purple', 
                        'light red', 'light yellow', 'logo', 'metal', 'opaque', 'orange', 'paper', 
                        'perforated', 'pink', 'plain', 'plastic', 'purple', 'rattan', 'red', 'stone', 
                        'striped', 'studded', 'text', 'translucent', 'transparent', 'velvet', 'white', 
                        'wood', 'wool', 'woven', 'yellow')

        begin_str = '<image>\n In the conversation below, you simply answer the category, subcategory name, and list of applied attributes based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            '"category:subcategory that is attributes". '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES) + '. '
        attr_str = 'Attributes Containing ' + ','.join(ATTR_CLASSES) 
        self.begin_str = begin_str + class_str + subclass_str + attr_str + '.\n'
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, use_bbox_text)
    
    def get_data_item(self, idx):
        data_info = self.get_data_info(idx)
        ann_info = self.get_ann_info(idx)

        img_path = os.path.join(self.img_prefix, data_info['filename'])
        image, image_size, image_token_len = self.process_image(img_path)

        gt_masks = []
        gt_bboxes = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)

            bbox = xywh2xyxy(ann['bbox']) # xywh to xyxy
            gt_bboxes.append(bbox)

            cat = self.coco.loadCats(ann['category_id'])
            gt_label = cat[0]['name']

            # Add attribute names if applicable
            atts = self.coco.loadAtts(ann['attribute_ids'])
            att_names = [att['name'] for att in atts]
            if att_names:
                gt_label = f"{gt_label} that is {','.join(att_names)}" 

            gt_labels.append(gt_label)

        data_item = dict(
            image = image,
            image_token_len=image_token_len,
            gt_bboxes = gt_bboxes,
            gt_masks = gt_masks,
            gt_labels = gt_labels
        )
        return data_item
    
class RefCOCO(CustomDataset):

    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 use_bbox_text=False,
                 ):


        self.begin_str = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.\n'
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, use_bbox_text)

    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue
            
            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
        
    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path = os.path.join(self.img_prefix, data_info['filename'])
        image, image_size, image_token_len = self.process_image(img_path)

        gt_masks = []
        gt_bboxes = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)

            bbox = xywh2xyxy(ann['bbox']) # xywh to xyxy
            gt_bboxes.append(bbox)
            
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(data_info['caption'])

        data_item = dict(
            image = image,
            image_token_len=image_token_len,
            gt_masks = gt_masks,
            gt_labels = gt_labels,
            gt_bboxes = gt_bboxes
        )

        return data_item

class RefCOCOP(RefCOCO):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_gt_per_img=15,
                 use_bbox_text=False,
                 ):
        self.begin_str = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image and its basic attibuts, you should not ' \
                         'give its position within the image.\n'                        
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, use_bbox_text=use_bbox_text)

if __name__ == '__main__':
    from types import SimpleNamespace

    import cv2
    import supervision as sv
    from transformers import AutoTokenizer
    from svg.model.multimodal_encoder.clip_encoder import get_image_processor

    tokenizer = AutoTokenizer.from_pretrained('exp/sg_stage2_ade_psg_vg_robin_gpt_sg_filtered_with_relation_region_shuffle_region_instruct_v1_vqa-qwen2.5-3b-instruct-lr1e-5-unfreeze_mm_vision_tower_bs64_epoch2_v5/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    # dataset = PartImagenet(
    #     tokenizer, data_args=data_args, 
    #     img_prefix="data/part_data/partImagenet/train",
    #     ann_file="data/part_data/partImagenet/partImagenet_train_format.json",
    # )

    dataset = PacoLVISPart(
        tokenizer, data_args,
        img_prefix='/net/nfs3.prior/jamesp/data/coco',
        ann_file='data/part_data/paco_lvis/paco_lvis_v1_train_format.json',
    )

    dataset[0]
    breakpoint()