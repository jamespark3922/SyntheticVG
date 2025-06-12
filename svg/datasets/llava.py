import copy
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from svg.train.train import preprocess, preprocess_multimodal

from .stage2_data import CustomDataset


def get_whole_image_mask(height, width) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.bool_)
    return mask.astype(np.float32)

class LlavaDataset(CustomDataset):
    def __init__(self,
                 tokenizer,
                 data_args=None,
                 ann_file=None,
                 img_prefix=None,
                 use_bbox_text=False,
                 sample=None,
                 ):
        self.begin_str = self.begin_str + self.region_str
        super().__init__(tokenizer, data_args, ann_file, img_prefix, use_bbox_text=use_bbox_text, sample=sample)
    
    def process_llava_text(self, s: str):
        s = s.replace('<image>\n','')
        s = s.replace('\n<image>','')
        s = s.replace('<','').replace('>','')
        return s

    def load_annotations(self, ann_file):

        data_infos = []
        
        ann_list = self.load_file(ann_file)

        for ann in tqdm(ann_list):
            if len(ann['conversations'])//2 ==0:
                continue
            qa_s = []

            if 'image' in ann:
                filename = ann['image']
                img_path = os.path.join(self.img_prefix, filename)
                # region_str = self.region_str
            else:
                img_path = None   
                # region_str = self.blank_str

            for i in range(len(ann['conversations'])//2):
                    
                question = ann['conversations'][i*2]['value']
                question = self.process_llava_text(question)
                # if i==0:
                #     question = region_str + question
                answer = ann['conversations'][i*2+1]['value']
                answer = self.process_llava_text(answer)
                # skip empty answer 
                if not answer:
                    print(f'Skipping empty answer for: {ann}')
                    continue
                qa_s.append({'from': 'human', 'value': question})         
                qa_s.append({'from': 'gpt', 'value': answer})
            
            if len(qa_s) > 0:
                data_infos.append(dict(
                    img_path = img_path,
                    convs = qa_s
                ))

        return data_infos
    
    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
            
        # Load Image
        if img_path is not None:
            image = self.load_image(img_path).convert('RGB')
        else:
            image = Image.new('RGB', (384,384))

        image, image_size, image_token_len = self.process_image(image)

        # process conversation
        convs = copy.deepcopy(data_info['convs'])
        convs[0]['value'] = self.begin_str + convs[0]['value']
        if debug:
            for conv in convs:
                print(conv['from'])
                print(conv['value'])
                print("=")
        sources = preprocess_multimodal(
            copy.deepcopy([convs]),
            self.data_args, 
            image_token_len
        )
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        
        data_dict['image'] = image
        data_dict['masks'] = self.get_whole_image_mask(image_size[1], image_size[0]) # None

        return data_dict