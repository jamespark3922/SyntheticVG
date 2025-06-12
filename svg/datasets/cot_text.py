import os
import random

import numpy as np
from tqdm import tqdm

from svg.constants import (
    PHRASE_END,
    PHRASE_END_QWEN2,
    PHRASE_START,
    PHRASE_START_QWEN2,
)

from .llava import LlavaDataset
from .prompts import GROUNDED_COT_GQA

SG_QUESTIONS = [
    'Generate a scene graph with relevant objects to answer the question using a single word or phrase.',
    'Create a scene graph highlighting objects critical for responding to the question using a single word or phrase.',
    'Develop a scene graph emphasizing objects essential for answering the question correctly.',
    'Build a scene graph featuring objects relevant to formulating an answer to the question.'
]

QUESTIONS = [
    'Answer the question using a single word or phrase.',
]


class CotTextDataset(LlavaDataset):
    '''
    Perform Scene Graph CoT via bounding box coordinates in text output. 
    Supports if image does not come with segmentation mask.
    '''

    BEGIN_STR = "<image>\nThis provides an overview of the picture and <region1> <mask><pos> highlighting the entire image.\n"

    def get_mask(self, w, h):
        return np.ones((1, h, w))

class GQACoTDataset(CotTextDataset):
    CLASSES = ('object',)

    def load_annotations(self, ann_file):

        st = "<ph_st>"
        ed = "<ph_ed>"

        phrase_st = PHRASE_START_QWEN2 if self.is_qwen2 else PHRASE_START
        phrase_ed = PHRASE_END_QWEN2 if self.is_qwen2 else PHRASE_END

        ann_list = self.load_file(ann_file)

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['imageId'])
            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            img_w = ann['img_w']
            img_h = ann['img_h']

            sg_s = []

            all_boxes = []

            for data in ann['data']:

                prompt = random.choice(GROUNDED_COT_GQA)
                question = f"{data['question']}\n{prompt}"

                # Parse CoT 
                sentence: str = data['cot']['value']
                boxes = data['cot']['boxes']
                boxes_seq: list[list[int]] = data['cot']['seq']
                assert len(boxes_seq) == sentence.count(ed)
                for idx, box_seq in enumerate(boxes_seq):
                    sentence = sentence.replace(st, phrase_st, 1)
                    boxes_list = [boxes[s] for s in box_seq]
                    bbox_text = self.bbox_to_text(boxes_list, img_h, img_w)
                    sentence = sentence.replace(ed, phrase_ed + ' ' + bbox_text, 1)
                assert sentence.count(st) == 0 and sentence.count(ed) == 0
                sg_s.append({'from': 'human', 'value': question})

                answer = f"{sentence}\nAnswer: {data['answer']}"
                sg_s.append({'from': 'gpt', 'value': answer})

                all_boxes += boxes

            data_infos.append(dict(
                img_path = img_path,
                boxes = all_boxes,
                convs = sg_s
            ))

        return data_infos


