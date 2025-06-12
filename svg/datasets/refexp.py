import copy
import json
import os
import random
from typing import Dict, List, Literal

import torch
from tqdm import tqdm

from svg.file_utils import open_gcs_or_local
from svg.train.train import preprocess, preprocess_multimodal

from .multi_region import MultiRegionDataset
from .prompts import GROUNDING_QUESTIONS, REGION_DESCRIPTION_QUESTIONS


def shuffle_regions_and_annotations(regions, annot):
    
    # Get a shuffled list of indices
    indices = list(range(len(regions)))
    random.shuffle(indices)

    # Use the shuffled indices to reorder the regions and annotation
    shuffled_regions = [regions[i] for i in indices]
    shuffled_annot = copy.deepcopy(annot)
    for annotation in shuffled_annot:
        annotation['region_index'] = indices.index(annotation['region_index'])

    return shuffled_regions, shuffled_annot

class RefExpDataset(MultiRegionDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 max_regions=30,
                 max_gt_per_img=10,
                 region_mode: Literal['box', 'segmentation', 'box_segmentation']='segmentation',
                 shuffle_regions: bool=False,
                 captioning_ratio: float=0,
                 use_bbox_text=False,
                 ):

        self.shuffle_regions = shuffle_regions
        self.captioning_ratio = captioning_ratio 
        assert 0 <= self.captioning_ratio < 1.0, "captioning_ratio should be a float between 0 and 1"
        super().__init__(tokenizer, data_args, ann_file, img_prefix, max_regions, max_gt_per_img,
                         region_mode=region_mode, use_bbox_text=use_bbox_text)
    
    def load_annotations(self, ann_file):
        """
        Load annotations from a given JSON file containing:

        - 'image': The filename of the image.
        - 'regions': A list of dictionaries, each representing a region in the image. Each region dictionary should contain:
            - 'xyxy': The coordinates of the bounding box for the region in the format [x1, y1, x2, y2].
            - 'segmentation': The segmentation of the region.
        - 'annot': A list of dictionaries, each representing an annotation for a region in the image.
            - 'caption': The caption for the region.
            - 'region_index': The index of the region in the 'regions' list.

        Args:
            ann_file (str): Path to the JSON file containing the annotations.

        Returns:
            data_infos (List[Dict]): data_infos to be used by __getitem__
        """

        with open_gcs_or_local(ann_file) as f:
            ann_list = json.load(f)
        
        data_infos = []
        for ann in tqdm(ann_list):

            img_path = os.path.join(self.img_prefix, ann['image'])

            # Sample max_gt_per_img
            annot: list[dict] = ann['annot']
            if self.is_train:
                annot: List[Dict] = random.sample(annot, len(annot))
            annot = annot[:self.max_gt_per_img]
            regions = ann['regions'][:self.max_regions]
            num_regions = len(regions)
            if num_regions == 0:
                print(f"Skipping {img_path} as it has no regions")
                continue
            h,w = regions[0]['segmentation']['size']
            
            # Process regions
            if self.use_bbox_text:
                for ann in annot:
                    bbox_text = self.bbox_to_text(ann['xyxy'], h, w)
                    ann['bbox_text'] = bbox_text
                boxes, segmentations = None, None
            else:
                # shuffle regions and adjust region_index in annot
                if self.is_train and self.shuffle_regions:
                    regions, annot = shuffle_regions_and_annotations(regions, annot)
                boxes, segmentations = self.process_regions(regions)
        
            # Create Conversation
            convs = self.generate_grounding_conversation(annot)
            if len(convs) > 0:
                data_infos.append(dict(
                        img_path = img_path,
                        boxes = boxes, 
                        segmentations=segmentations,
                        convs = convs)
                )
            
            convs = self.generate_grounded_captioning_conversation(annot)
            if len(convs) > 0:
                data_infos.append(dict(
                        img_path = img_path,
                        boxes = boxes, 
                        segmentations=segmentations,
                        convs = convs)
                )
 
        return data_infos
    
    def generate_grounding_conversation(self, annot):
        convs = []
        for idx,d in enumerate(annot):
            region_index = d['region_index']
            caption = d['caption']

            # Grounding
            q = random.choice(GROUNDING_QUESTIONS) + caption
            if self.use_bbox_text:
                a = d['bbox_text']
            else:
                a = f"region{region_index+1}"
            
            convs.append({'from': 'human', 'value': q})
            convs.append({'from': 'gpt', 'value': a})
        
        return convs
    
    def generate_grounded_captioning_conversation(self, annot,):
        convs = []
        for idx,d in enumerate(annot):
            region_index = d['region_index']
            caption = d['caption']

            # Grounded Captioning
            q = random.choice(REGION_DESCRIPTION_QUESTIONS)
            if self.use_bbox_text:
                region_query = d['bbox_text']
            else:
                region_query = f"region{region_index+1}"
            q = q.replace('<region>', region_query)
            a = caption
            
            convs.append({'from': 'human', 'value': q})
            convs.append({'from': 'gpt', 'value': a})
        
        return convs
    
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
        convs = copy.deepcopy(convs)

        # Add image and region prefix
        if self.use_bbox_text:
            pred_masks = self.get_whole_image_mask(h, w)
            convs[0]['value'] = self.begin_str + self.region_str + convs[0]['value']
        else:
            pred_masks= torch.Tensor(self.create_masks(boxes, segmentations, h, w))
            num_objects = len(pred_masks)
            region_string = self.get_region_string(num_objects)
            convs[0]['value'] = self.begin_str + region_string + convs[0]['value']

        sources = preprocess_multimodal(
            copy.deepcopy([convs]),
            self.data_args, image_token_len
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
        data_dict['masks'] = pred_masks

        return data_dict
