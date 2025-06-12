import copy
import logging
import random
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import torch

from svg.train.train import preprocess, preprocess_multimodal

from .prompts import REF_WAY_NUM
from .stage2_data import CustomDataset, xywh2xyxy

class RegionMode(str, Enum):
    BOX = 'box'
    SEGMENTATION = 'segmentation'
    BOX_SEGMENTATION = 'box_segmentation'

class RegionAugmentationType(str, Enum):
    SHUFFLE = 'shuffle'
    TRUNCATE = 'truncate'
    SHUFFLE_AND_TRUNCATE = 'shuffle_and_truncate'
    SORT_BY_SIZE = 'sort_by_size'
    NONE = 'none'

class RegionAugmentation:
    ''' Class for applying data augmentation to regions in training'''

    def __init__(self, augmentation_type: str | RegionAugmentationType, min_region_keep_ratio=0.7):

        # Set the augmentation type
        if isinstance(augmentation_type, str):
            try:
                augmentation_type = RegionAugmentationType(augmentation_type)
            except ValueError:
                logging.warning(f"Invalid region augmentation type: {augmentation_type}")
                self.augmentation_type = None
                return

        # Check if value is a valid enum member
        if not isinstance(augmentation_type, RegionAugmentationType):
            logging.warning(f"Invalid region augmentation type: {augmentation_type}")
            self.augmentation_type = None
        else:
            self.augmentation_type = augmentation_type

        self.min_region_keep_ratio = min_region_keep_ratio
        self.max_regions = None
    
    def __repr__(self):
        return f"RegionAugmentation(augmentation_type={self.augmentation_type}, min_region_keep_ratio={self.min_region_keep_ratio})"
    
    def __call__(self, regions: list[dict], region_mapping: dict[str, int]) -> tuple[list[dict], dict[str, int]]:
        if self.augmentation_type == RegionAugmentationType.SHUFFLE:
            return self.shuffle_regions_and_update_mapping(regions, region_mapping)
        elif self.augmentation_type == RegionAugmentationType.TRUNCATE:
            return self.truncate_regions(regions, region_mapping, n=self.max_regions)
        elif self.augmentation_type == RegionAugmentationType.SHUFFLE_AND_TRUNCATE:
            regions, region_mapping = self.shuffle_regions_and_update_mapping(regions, region_mapping)
            return self.truncate_regions(regions, region_mapping, n=self.max_regions)
        elif self.augmentation_type == RegionAugmentationType.SORT_BY_SIZE:
            return self.sort_regions_by_size(regions, region_mapping)
        else:
            return regions, region_mapping
    
    def _update_regions_and_mapping(self, regions, region_mapping, new_indices):
        """Update regions and region mapping based on new indices ordering."""
        updated_regions = [regions[i] for i in new_indices]
        index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(new_indices)}
        updated_region_mapping = {obj_id: index_mapping[region_idx] 
                                for obj_id, region_idx in region_mapping.items() 
                                if region_idx in index_mapping}
        return updated_regions, updated_region_mapping

    def shuffle_regions_and_update_mapping(self, regions: List, region_mapping: Dict[str, int]) -> Tuple[List, Dict[str, int]]:
        """
        Shuffles regions list and updates region_mapping with new indices.
        Randomizes region order for training and debiasing region index.

        Args:
            regions (List): List of region dictionaries.
            region_mapping (Dict[str, int]): Mapping from object IDs to region indices.

        Returns:
            Tuple[List, Dict[str, int]]: 
                - Shuffled regions list
                - Updated region_mapping with new indices

        Example:    
            >>> regions = [
            ...     {'object_id': '0', 'segmentation': {...}},  # Region at index 0
            ...     {'object_id': '1', 'segmentation': {...}},  # Region at index 1
            ...     {'object_id': '2', 'segmentation': {...}},  # Region at index 2 
            ...     {'object_id': '3', 'segmentation': {...}},  # Region at index 3
            ...     {'object_id': '4', 'segmentation': {...}}   # Region at index 4
            ... ]
            >>> # Only some object IDs are used in the conversation and need mapping
            >>> region_mapping = {'0': 0, '1': 1, '2': 2, '4': 4}  # Maps object_id -> index in regions list
            >>> 
            >>> # After shuffle, regions might be rearranged to [4,3,0,1,2]
            >>> shuffled_regions, updated_region_mapping = shuffle_regions_and_update_mapping(regions, region_mapping)
            >>> 
            >>> # Region with object_id '4' is now at index 0, '0' at index 2, etc.
            >>> updated_region_mapping
            >>> {'0': 2, '1': 3, '2': 4, '4': 0}
        """
        # Create a list of indices corresponding to each region
        shuffled_indices = list(range(len(regions)))
        random.shuffle(shuffled_indices)

        shuffled_regions, updated_region_mapping = self._update_regions_and_mapping(regions, region_mapping, shuffled_indices)
        return shuffled_regions, updated_region_mapping
    
    def truncate_regions(self, regions: List, region_mapping: Dict[str, int], n: int = None) -> Tuple[List, Dict[str, int]]:
        """
        Truncates the regions list and updates the region_mapping to reflect the new indices of the regions.
        Useful for being robust to incomplete segmentations.

        Args:
            regions (List): The list of regions to be cropped.
            region_mapping (Dict[str, int]): The mapping from object IDs to region index.
            n: The number of regions to keep. (default: None)
                - If None, randomly selects a number of regions to keep.
        Returns:
            Tuple[List, Dict[str, int]]: The cropped list of regions and the updated region mapping.
        """

        # Crop the regions list and update the region_mapping to reflect the new indices of the regions.
        if n is None:
            # Randomly select a number of regions to keep
            min_regions = max(1, int(self.min_region_keep_ratio * len(regions)))
            n = random.randint(min_regions, len(regions))
        num_regions = min(n, len(regions))
        regions_tr = regions[:num_regions]
        region_mapping_tr = {k: v for k, v in region_mapping.items() if v < num_regions}

        return regions_tr, region_mapping_tr
    
    def sort_regions_by_size(self, regions: List, region_mapping: Dict[str, int]) -> Tuple[List, Dict[str, int]]:
        """
        Sorts the regions list by size and updates the region_mapping to reflect the new indices of the regions.
        Useful for training and debiasing region index by sorting the regions by size.

        Args:
            regions (List): The list of regions to be sorted by size.
            region_mapping (Dict[str, int]): The mapping from object IDs to region index.

        Returns:
            Tuple[List, Dict[str, int]]: The sorted list of regions and the updated region mapping.
        """

        # Sort the regions list by size and update the region_mapping to reflect the new indices of the regions.
        if len(regions) == 0:
            return regions, region_mapping
        
        def get_region_area(region):
            if 'area' in region:
                return region['area']
            if 'xyxy' in region:
                return (region['xyxy'][2] - region['xyxy'][0]) * (region['xyxy'][3] - region['xyxy'][1])
            if 'bbox' in region:
                return region['bbox'][2] * region['bbox'][3]
            raise ValueError("Region must have 'area' or 'xyxy' or 'bbox' attribute to sort by size")
        
        region_indices = list(range(len(regions)))
        region_indices = sorted(region_indices, key=lambda x: get_region_area(regions[x]), reverse=True)

        sorted_regions, updated_region_mapping = self._update_regions_and_mapping(regions, region_mapping, region_indices)
        
        return sorted_regions, updated_region_mapping

class MultiRegionDataset(CustomDataset):
    
    def __init__(self,
                tokenizer=None,
                data_args=None,
                ann_file=None,
                img_prefix=None,
                max_regions=30,
                max_gt_per_img=20,
                is_train=True,
                region_augmentation: RegionAugmentationType = RegionAugmentationType.SHUFFLE,
                region_mode: RegionMode = RegionMode.SEGMENTATION,
                min_region_keep_ratio=0.7,
                use_bbox_text=False,
                sample=None,
    ):

        self.max_regions = max_regions
        self.is_train = is_train
        self.region_mode = RegionMode(region_mode)
        self.region_augmentation = RegionAugmentation(
            region_augmentation, 
            min_region_keep_ratio=min_region_keep_ratio
        )
        self.use_box = region_mode == 'box'
        logging.info(f"Region mode: {region_mode}")
        logging.info(f"Region augmentation: {region_augmentation}")
        logging.info(f"Use bbox text: {use_bbox_text}")
        logging.info(f"Max regions: {max_regions}")
        
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_gt_per_img, 
                         use_bbox_text=use_bbox_text, sample=sample)
    
    def load_annotations(self, ann_file):
        """
        Loads annotations from a given annotation file.
        Should be implemented in the subclass.

        Args:
            ann_file (str): Path to the annotation file.

        Returns:
            list of dict: A list of dictionaries, where each dictionary contains:
                - 'img_path' (str): The path to the image file.
                - 'boxes' (list of lists): The bounding boxes for objects in the image. Each bounding box is represented as a list of four integers [x, y, width, height].
                - 'segmentations' (list of lists): The segmentation masks for objects in the image. Each mask is represented as a list of coordinates [[x1, y1], [x2, y2], ...].
                - 'convs' (list of dicts): A list of dictionaries, each representing a conversation. Each conversation dictionary contains a 'value' key, which is a string representing the conversation.

        Example:
        [
            {
                'img_path': '/path/to/image.jpg',
                'boxes': [[10, 20, 50, 50], [70, 80, 100, 100]],
                'segmentations': [[[10, 20], [30, 40], [50, 60]], [[70, 80], [90, 100], [110, 120]]],
                'convs': [{'value': 'Hello, how are you?'}]
            },
        ]
        """
        raise NotImplementedError
    
    @staticmethod
    def process_regions(regions: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ Convert detected regions into boxes and segmentations."""

        # Extract boxes from regions
        if len(regions) > 0 and 'xyxy' in regions[0]:
            boxes: list[np.ndarray] = [region['xyxy'] for region in regions]
        else:
            boxes: list[np.ndarray] = [xywh2xyxy(region['bbox']) for region in regions]

        # Extract segmentations if available
        if len(regions) > 0 and 'segmentation' in regions[0]:
            segmentations: list[np.ndarray] = [region['segmentation'] for region in regions]
            if len(boxes) != len(segmentations):
                raise ValueError(
                    f"Number of boxes ({len(boxes)}) and segmentations ({len(segmentations)}) must match"
                )
            return boxes, segmentations
        
        return boxes, None
   
    @staticmethod 
    def get_region_id(id: str, region_mapping: Dict[str, int]) -> int:
        """
        Convert an object ID to its corresponding 1-indexed region ID for use in conversation.
        
        This method is important for creating human-readable region references in conversations.
        While regions are stored with 0-indexed positions internally, they're presented to users
        with 1-indexed IDs (region1, region2, etc.) for better readability.
        
        Args:
            id (str): The object ID to convert. This is a string identifier from the original 
                      scene graph or annotation that uniquely identifies an object.
            region_mapping (Dict[str, int]): A mapping from object IDs to their 0-indexed 
                                            positions in the regions list.
        
        Returns:
            int: The 1-indexed region ID to use in conversation (e.g., for "region3").
            
        Example:
            If region_mapping = {'object_0': 2, 'object_1': 0, 'object_2': 1}
            get_region_id('object_0', region_mapping) returns 3 (for "region3" in conversation)
        """
        return int(region_mapping[id]) + 1  # Convert from 0-indexed to 1-indexed
    
    def get_region_string(self, n: int, bbox_texts: list[str]=None):
        """
        Get a prefix string describing number of regions, and mask and bounding box (optional) info for each region.

        Args:
            n (int): The number of regions.
            bbox_texts (list[str], optional): A list of bbox texts saying xyxy coordinates, same size as `n`. Defaults to None.

        Returns:
            str: A prefix string to use.
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

        # ref_prefix = random.choice(REF_WAY)
        ref_prefix = random.choice(REF_WAY_NUM).format(n)
        region_string = ref_prefix.replace('<region>', ref_string)
        region_string = region_string

        return region_string
    
    def textify_region(self, region_id: int) -> str:
        ''' Returns a string representation of the region id.'''
        return 'region' + str(region_id+1)
    
    def create_masks(self, boxes: list, segmentations: list, h, w) -> np.ndarray:
        """
        Args:
            boxes (list): the ground truth bounding boxes
            segmentations (list): list of ground truth segmentation masks
            h (int): the height of the image
            w (int): the width of the image

        Returns:
            np.ndarray: returned segmentation mask, one for each object in the image.
        """

        # pred_masks = np.zeros((len(segmentations), h, w))
        # for i in range(len(pred_masks)):

        #     pred_mask = None
        #     bbox = boxes[i]
        #     mask = segmentations[i]
        #     pred_mask = self._create_single_mask(bbox, mask, h, w)
        #     pred_masks[i] = pred_mask
            
        # return pred_masks

        pred_masks = []
        for box, mask in zip(boxes, segmentations):
            pred_mask: np.ndarray = self._create_single_mask(box, mask, h, w)
            pred_masks.append(pred_mask)
        
        return np.array(pred_masks)

    
    def _create_single_mask(self, bbox: list | np.ndarray, mask, h, w) -> np.ndarray:
        """
        Creates a single mask based on the region mode.

        Args:
            bbox (np.ndarray): the bounding box for the object
            mask : the segmentation mask for the object. can be np array, or coco rle format.
            h (int): the height of the image
            w (int): the width of the image

        Returns:
            np.ndarray: the created mask for the object
        """

        if not all([isinstance(x, (int, float)) for x in [h, w]]) or h <= 0 or w <= 0:
            raise ValueError("Invalid dimensions: height and width must be positive numbers")
        
        if self.region_mode == RegionMode.BOX:
            return self.bboxToMask(bbox, h, w)
            
        if self.region_mode == RegionMode.SEGMENTATION:
            return self.annToMask(mask, h, w)
            
        if self.region_mode == RegionMode.BOX_SEGMENTATION:
            internal_mask = None if mask is None else self.annToMask(mask, h, w)
            # If mask is None, we will use bbox as mask
            return self.bboxToMaskWithBorder(
                bbox, h, w, 
                border_thickness=3, 
                internal_mask=internal_mask
            )
            
        raise ValueError(f"Unsupported region mode: {self.region_mode}")

    def __getitem__(self, i, debug=False):
        """
        Retrieves an item from the dataset at a given index.

        Args:
            i (int): The index of the item to retrieve.
            debug (bool, optional): If True, provides additional debug information for the conversation. Defaults to False.

        Returns:
            dict: A dictionary containing the processed image, its associated masks, and conversation tokens.

        The expected format of `data_infos` is a list of dictionaries, where each dictionary contains:
            - 'img_path': The path to the image file.
            - 'boxes': Absolute xyxy bounding boxes for objects in the image.
            - 'segmentations': The segmentation masks for objects in the image.
            - 'convs': A list of dictionaries, each representing a conversation.
        """
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        boxes = data_info['boxes']
        segmentations = data_info['segmentations']
        convs: List[Dict] = data_info['convs']

        image, image_size, image_token_len = self.process_image(img_path)
        w, h = image_size

        # Load masks as bbox, segm, or bbox + segm
        pred_masks: np.ndarray = self.create_masks(boxes, segmentations, h, w)

        convs = copy.deepcopy(convs)

        # Add image and region prefix
        num_objects = len(pred_masks)
        if self.use_bbox_text:
            if boxes is None:
                bbox_texts = [self.mask_to_bbox_text(mask) for mask in pred_masks]
            else:
                bbox_texts = [self.bbox_to_text(bbox, h, w) for bbox in boxes]
            region_string = self.get_region_string(num_objects, bbox_texts)
        else:
            region_string = self.get_region_string(num_objects)
        convs[0]['value'] = self.begin_str + region_string + convs[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([convs]),
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
            has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image
        data_dict['masks'] = torch.Tensor(pred_masks)

        return data_dict