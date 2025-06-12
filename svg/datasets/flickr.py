import os
import random

from tqdm import tqdm

from svg.constants import (
    PHRASE_END,
    PHRASE_END_QWEN2,
    PHRASE_START,
    PHRASE_START_QWEN2,
)

from .llava import LlavaDataset
from .prompts import GROUNDED_DESCRIPTION_QUESTIONS


class FlickrEntitiesDataset(LlavaDataset):
    
    def load_annotations(self, ann_file):

        st = "<ph_st>"
        ed = "<ph_ed>"

        phrase_st = PHRASE_START_QWEN2 if self.is_qwen2 else PHRASE_START
        phrase_ed = PHRASE_END_QWEN2 if self.is_qwen2 else PHRASE_END

        ann_list = self.load_file(ann_file)

        data_infos = []
        for idx,ann in enumerate(tqdm(ann_list)):

            image_id = str(ann['image_id'])
            img_path = os.path.join(self.img_prefix, image_id+'.jpg')

            img_w = ann['img_w']
            img_h = ann['img_h']

            question = random.choice(GROUNDED_DESCRIPTION_QUESTIONS)
            sentence: str = ann['sentence']
            boxes = ann['boxes']
            boxes_seq: list[list[int]] = ann["boxes_seq"] 

            assert len(boxes_seq) == sentence.count(ed)
            for idx, box_seq in enumerate(boxes_seq):
                sentence = sentence.replace(st, phrase_st, 1)
                boxes_list = [boxes[s] for s in box_seq]
                bbox_text = self.bbox_to_text(boxes_list, img_h, img_w)
                sentence = sentence.replace(ed, phrase_ed + ' ' + bbox_text, 1)
            
            assert sentence.count(st) == 0 and sentence.count(ed) == 0

            sg_s = []
            sg_s.append({'from': 'human', 'value': question})
            sg_s.append({'from': 'gpt', 'value': sentence})

            data_infos.append(dict(
                img_path = img_path,
                boxes = boxes,
                convs = sg_s
            ))
            
        
        return data_infos

