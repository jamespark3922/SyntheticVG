import json
import os
import random
from tqdm import tqdm

import numpy as np
import torch

from .llava import LlavaDataset

GROUNDING_QUESTIONS = [
    'Please provide the region that best answers this question: ',
    'What is the region that best answers this question: ',
    'Which area best answers this question: ',
    'Can you identify the region for this question: ',
    'What region is being referred to in this question: ',
]

class PointQADataset(LlavaDataset):
    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            img_prefix=None,
            use_bbox_text=True,
            use_point_text=False,
    ):
        
        self.use_point_text = use_point_text
        super().__init__(tokenizer, data_args, ann_file, img_prefix, 
                         use_bbox_text=use_bbox_text)
    
    def get_qa_prompt(self, prompt):
        return f"{prompt}\nAnswer the question using a single word or phrase."

class PointQALocalDataset(PointQADataset):
    CLASSES = ('object',)
    
    def load_annotations(self, ann_file):

        assert self.use_bbox_text or self.use_point_text, 'At least one of use_bbox_text or use_point_text should be True'
        ann_list = []
        with open(ann_file, 'r') as f:
            for line in f:  
                ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['genome_id'])

            img_path = os.path.join(self.img_prefix, image_id+'.jpg')
            w = ann['img_w']
            h = ann['img_h']
    
            # segmentations = boxes
            question = ann['question']
            answer = ann['answer']
            bbox = ann['bbox'] # xyxy
            point = ann['point'] # xy

            # bbox prompt
            if self.use_bbox_text:
                sg_s = []
                prompt = f"{question} {self.bbox_to_text(bbox, h, w)}"
                prompt = self.get_qa_prompt(prompt)
                sg_s.append({'from': 'human', 'value': prompt})
                sg_s.append({'from': 'gpt', 'value': answer})

                data_infos.append(dict(
                    img_path = img_path,
                    bbox=bbox,
                    point=point,
                    convs = sg_s)
                )

            # point prompt
            if self.use_point_text:
                sg_s = []
                prompt = f"{question} {self.point_to_text(point, h, w)}"
                prompt = self.get_qa_prompt(prompt)
                sg_s.append({'from': 'human', 'value': prompt})
                sg_s.append({'from': 'gpt', 'value': answer})

                data_infos.append(dict(
                    img_path = img_path,
                    bbox=bbox,
                    point=point,
                    convs = sg_s)
                )
            
        return data_infos


class PointQATwiceDataset(PointQADataset):
    CLASSES = ('object',)

    def load_annotations(self, ann_file):

        ann_list = []
        with open(ann_file, 'r') as f:
            for line in f:  
                ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['genome_id'])

            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            # No need for bbox or point
            question = ann['obj_question']
            answer = ann['answer']
            bbox = ann['bbox'] # xyxy
            point = ann['point'] # xy

            sg_s = []
            prompt = self.get_qa_prompt(question)
            sg_s.append({'from': 'human', 'value': prompt})
            sg_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                boxes=[bbox],
                point=point,
                convs = sg_s)
            )

            question = ann['general_question'] # 'How many of these are in the picture?
            w = ann['img_w']
            h = ann['img_h']
            if self.use_bbox_text:
                sg_s = []
                prompt = f"{question} {self.bbox_to_text(bbox, h, w)}"
                prompt = self.get_qa_prompt(prompt)
                sg_s.append({'from': 'human', 'value': prompt})
                sg_s.append({'from': 'gpt', 'value': answer})

                data_infos.append(dict(
                    img_path = img_path,
                    boxes=[bbox],
                    point=point,
                    convs = sg_s)
                )

            # point prompt
            if self.use_point_text:
                sg_s = []
                prompt = f"{question} {self.point_to_text(point, h, w)}"
                prompt = self.get_qa_prompt(prompt)
                sg_s.append({'from': 'human', 'value': prompt})
                sg_s.append({'from': 'gpt', 'value': answer})

                data_infos.append(dict(
                    img_path = img_path,
                    boxes=[bbox],
                    point=point,
                    convs = sg_s)
                )
        
        return data_infos

class V7WPointQADataset(LlavaDataset):
    CLASSES = ('object',)

    def __init__(
            self,
            tokenizer,
            data_args=None,
            ann_file=None,
            img_prefix=None,
            is_train=True,
    ):
        # Always use box mode for PointQA
        self.is_train = is_train
        super().__init__(tokenizer, data_args, ann_file, img_prefix, 
                         use_bbox_text=True)
    
    def get_mc_prompt(self, question: str, bbox_texts: list[str]):
        gq = random.choice(GROUNDING_QUESTIONS) if self.is_train else GROUNDING_QUESTIONS[0]
        prompt = gq + question
        
        box_options = ''
        for idx, bbox_text in enumerate(bbox_texts):
            box_options += f"{chr(ord('A')+idx)}. {bbox_text}\n"
        prompt = f"{prompt}\n{box_options}Answer with the option's letter from the given choices directly."

        return prompt

    def load_annotations(self, ann_file):

        ann_list = []
        with open(ann_file, 'r') as f:
            for line in f:  
                ann_list.append(json.loads(line))

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            img_path = os.path.join(self.img_prefix, ann['file_path'])
            img_h = ann['img_h']
            img_w = ann['img_w']
            boxes = ann['candidates']
            gt_idx = 3 # seems last one is the GT box
            if self.is_train: # shuffle box choices
                box_indices = list(range(len(boxes)))
                random.shuffle(box_indices)
                boxes = [boxes[idx] for idx in box_indices]
                gt_idx = box_indices.index(3) 
            bbox_texts = [self.bbox_to_text(box, img_h, img_w) for box in boxes]
            question = ann['question']
            answer: str = chr(ord('A')+gt_idx)

            sg_s = []
            prompt = self.get_mc_prompt(question, bbox_texts)
            sg_s.append({'from': 'human', 'value': prompt})
            sg_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                boxes=boxes,
                convs = sg_s)
            )
        
        return data_infos
