"""
This code is largely based on https://github.com/jshilong/GPT4RoI/blob/main/gpt4roi/datasets/vcr.py
"""
import os
from tqdm import tqdm

from svg.datasets.llava import LlavaDataset

ANSWERS = ['no', 'yes']

def get_text(datum: dict):
    if datum['caption'].split().count('is') == 1:
        text = f"Is the {datum['subj']} {datum['relation']} the {datum['obj']}?"
    else:
        relation = datum['relation']
        if relation == 'has as a part':
            text = f"Is the {datum['subj']} part of the {datum['obj']}?"
        else:
            if relation[-1] == 's':
                relation = relation[:-1]
            text = f"Does the {datum['subj']} {relation} the {datum['obj']}?"
    return text

class VSRDataset(LlavaDataset):

    prompt_postfix = "Answer directly with 'yes' or 'no'."

    def load_annotations(self, ann_file):

        data_infos = []
        
        ann_list = self.load_file(ann_file)

        for ann in tqdm(ann_list):

            img_path = os.path.join(self.img_prefix, ann['image'])
            question = get_text(ann)
            question = f"{question} {self.prompt_postfix}"
            answer = ANSWERS[ann['label']]

            qa_s = []
            qa_s.append({'from': 'human', 'value': question})         
            qa_s.append({'from': 'gpt', 'value': answer})
            
            if len(qa_s) > 0:
                data_infos.append(dict(
                    img_path = img_path,
                    convs = qa_s
                ))

        return data_infos

if __name__ == '__main__':
    from types import SimpleNamespace

    import cv2
    import supervision as sv
    from transformers import AutoTokenizer
    from svg.model.multimodal_encoder.clip_encoder import get_image_processor

    tokenizer = AutoTokenizer.from_pretrained('exp/sg_stage2_ade_psg_vg_robin_gpt_sg_filtered_with_relation_region_shuffle_region_instruct_v1_vqa-qwen2.5-3b-instruct-lr1e-5-unfreeze_mm_vision_tower_bs64_epoch2_v5/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = VSRDataset(
        tokenizer, 
        data_args=data_args, 
        ann_file="/net/nfs3.prior/jamesp/code/SynthVG/data/vsr/zeroshot_train.jsonl",
        img_prefix="/net/nfs3.prior/jamesp/data/coco/train2017/",
    )
    print(len(dataset))
    breakpoint()
    print(dataset.data_infos[1]['convs'][1]['value'])