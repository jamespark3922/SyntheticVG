
import json
import os
from urllib.parse import urlparse
from PIL import Image
import cv2
import numpy as np
import requests
import argparse

from segment_anything import sam_model_registry

from svg.pipeline.region_proposal.region_generator import SamGroundingDinoRegionGenerator
from svg.pipeline.grounding.grounding_dino import GroundingDinoSAM
from svg.pipeline.captioning.gpt4o import GPT4Captioner
from svg.pipeline.robin import RobinPipeline
from svg.draw_utils import  visualize_masks

def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def load_image(image_path, timeout=10) -> Image.Image:
    """Load image from local path or URL.
    
    Args:
        image_path: Local file path or HTTP/HTTPS URL
        timeout: Timeout in seconds for URL requests
    Returns:
        PIL Image object
    Raises:
        ValueError: If path is invalid or image can't be loaded
    """
    # Check if URL
    result = urlparse(image_path)
    is_url = all([result.scheme in ('http', 'https'), result.netloc])
    
    if is_url:
        response = requests.get(image_path, stream=True, timeout=timeout)
        response.raise_for_status()
        image = Image.open(response.raw)
    else:
        if not os.path.exists(image_path):
            raise ValueError(f"Local file not found: {image_path}")
        image = Image.open(image_path)
    
    # Validate it's actually an image
    return image.convert('RGB')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='robin scene graph predict',)
    parser.add_argument('--image', type=str, help='image path', required=True)
    parser.add_argument('--robin_ckpt', default='exp/sg_stage2_ade_psg_vg_robin_gpt_sg_filtered_with_relation_region_shuffle_region_instruct_v1_vqa-qwen2.5-3b-instruct-lr1e-5-unfreeze_mm_vision_tower_bs64_epoch2_v5/', 
                        help='path to robin')
    parser.add_argument('--sam_ckpt', default='sam_vit_h_4b8939.pth', help='SAM checkpoint')
    parser.add_argument('--grounding_dino_ckpt', 
                        default='IDEA-Research/grounding-dino-base', 
                        help='path to robin')
    args = parser.parse_args()

    image = load_image(args.image)
    device = 'cuda'

    # SAM model for region proposal
    print('Loading SAM model...')
    sam_model = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).to(device)

    # Optional: grounding_dino + gpt4o captioner for additional region grounding
    print('Loading GroundingDino model...')
    grounding_model = GroundingDinoSAM(
        args.grounding_dino_ckpt,
        sam_model, 
        device
    )
    captioner = GPT4Captioner(api_key=None)
    region_generator = SamGroundingDinoRegionGenerator(
        sam_model=sam_model,
        grounding_model=grounding_model, # None if not using
        captioner=captioner
    )

    # Generate regions
    print('Generating regions...')
    regions: list[dict] = region_generator.generate_regions(image, region_mode='merged') # [N] list of dict
    print(f'Generated {len(regions)} regions')
    with open('scene_graph_regions.json', 'w') as f:
        json.dump(regions, f, indent=4)
    print('Regions saved to scene_graph_regions.json')

    # Generate scene graph
    print('Loading Robin...')
    robin = RobinPipeline(args.robin_ckpt, device=device)
    scene_graph, _ = robin.generate_scene_graph(image, regions)

    image_rgb = np.array(image)
    image_with_masks: np.ndarray = visualize_masks(
        image_rgb, regions, 
        draw_bbox=True, draw_mask = True, draw_polygon=False,
        white_padding=50
        )
    # convert to BGR
    image_with_masks = cv2.cvtColor(image_with_masks, cv2.COLOR_RGB2BGR)
    cv2.imwrite('scene_graph.jpg', image_with_masks)
    
    # Align with regions
    region_objects = []
    assert len(regions) == len(scene_graph['objects'])
    for idx, (region, obj) in enumerate(zip(regions, scene_graph['objects'])):
        bbox = np.array(region['bbox'], dtype=int).tolist() # xywh
        bbox = xywh2xyxy(bbox)
        region_objects.append({
            'id': idx,
            'bbox': bbox,
            'description': obj,
        })
    scene_graph['objects'] = region_objects
    with open('scene_graph.json', 'w') as f:
        json.dump(scene_graph, f, indent=4)
    print('Scene graph saved to scene_graph.json')