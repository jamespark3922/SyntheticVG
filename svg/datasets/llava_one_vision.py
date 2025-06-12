import copy
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets

from svg.train.train import preprocess, preprocess_multimodal


from .llava import LlavaDataset


class LlavaOneVisionDataset(LlavaDataset):
    
    def load_annotations(self, ann_file):
        '''
        >>> dataset
        Dataset({
                features: ['id', 'image', 'conversations', 'data_source'],
                num_rows: 2152
        })
        '''
        dataset = load_dataset(self.img_prefix, ann_file, split='train')
        return dataset
    
    @staticmethod
    def sample_data(dataset: Dataset, target_size: int):
        """
        Sample a Hugging Face dataset to reach a target size
        
        Args:
            dataset (Dataset): Hugging Face dataset
            target_size (int): Target dataset size
        
        Returns:
            Dataset: Sampled dataset
        """

        dataset_size = len(dataset)
        
        if target_size <= 0:
            return Dataset.from_dict({})
        
        if target_size <= dataset_size:
            # If we need fewer examples than we have, just sample randomly
            indices = random.sample(range(dataset_size), target_size)
            return dataset.select(indices)
        else:
            # If we need more examples, repeat the dataset and sample for remainder
            n_repeats = target_size // dataset_size
            remainder = target_size % dataset_size
            
            # Create n complete copies
            datasets_to_concat = [dataset] * n_repeats
            
            # Add the remainder using random sampling if needed
            if remainder > 0:
                remainder_indices = random.sample(range(dataset_size), remainder)
                remainder_dataset = dataset.select(remainder_indices)
                datasets_to_concat.append(remainder_dataset)
            
            # Concatenate all parts
            return concatenate_datasets(datasets_to_concat)
        
    
    def __getitem__(self, i, debug=False):
        data_info = self.data_infos[i]
            
        # Load Image
        image = data_info["image"]
        if image is None:
            image = Image.new('RGB', (384,384))
        image, image_size, image_token_len = self.process_image(image)

        # process conversation
        convs = copy.deepcopy(data_info['conversations'])
        for conv in convs:
            conv['value'] = self.process_llava_text(conv['value']) # strips image token
        convs[0]['value'] = self.begin_str + convs[0]['value'] # add begin token with image token
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

if __name__ == '__main__':
    from types import SimpleNamespace

    import cv2
    import supervision as sv
    from transformers import AutoTokenizer
    from svg.model.multimodal_encoder.clip_encoder import get_image_processor

    def draw_segmentation(idx: int, output_image='llava_one_vision.jpg'):
        info = dataset.data_infos[idx]
        data = dataset.__getitem__(idx, debug=True)
        image = info['image']
        im = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        mask = np.array(data['masks'], dtype=bool)
        ids = np.array(range(1, len(mask)+1))
        labels = [f"[{idx}]" for idx in ids]

        annotated_image = im.copy()
        cv2.imwrite(output_image, annotated_image)

    tokenizer = AutoTokenizer.from_pretrained('exp/sg_stage2_ade_psg_vg_robin_gpt_sg_filtered_with_relation_region_shuffle_region_instruct_v1_vqa-qwen2.5-3b-instruct-lr1e-5-unfreeze_mm_vision_tower_bs64_epoch2_v5/')
    image_processor = get_image_processor(img_size=512)
    data_args = SimpleNamespace(image_processor=image_processor, mm_use_im_start_end=False, image_aspect_ratio='pad')
    dataset = LlavaOneVisionDataset(
        tokenizer, data_args=data_args, 
        img_prefix="/net/nfs3.prior/jamesp/data/LLaVA-OneVision-Data",
        # ann_file="VizWiz(MathV360K)",
        ann_file="clevr(cauldron,llava_format)",
        sample=0.5
    )
    print(len(dataset))
    draw_segmentation(0)
    breakpoint()
    print(dataset.data_infos[1]['convs'][1]['value'])
