# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import os
import argparse
from PIL import Image
import pickle
import json
from tqdm import tqdm

# seem
# from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
# from seem.utils.distributed import init_distributed as init_distributed_seem
# from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import inference_seem_pano, som_w_gt

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model as build_model_semantic_sam
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

from azfuse import File

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto

from merge_som_proposals import compress_seg, resize_region 

SAM_CKPT="/net/nfs.cirrascale/mosaic/jamesp/code/unified-sg/segment-anything/models/sam_vit_h_4b8939.pth"
ROOT_DIR= os.path.dirname(os.path.abspath(__file__))

def build_som_model(model_name, device='cuda'):
    '''
    build model
    '''
    if model_name == 'semantic-sam':
        semsam_cfg = os.path.join(ROOT_DIR, "configs/semantic_sam_only_sa-1b_swinL.yaml")
        opt_semsam = load_opt_from_config_file(semsam_cfg)
        semsam_ckpt = os.path.join(ROOT_DIR, "swinl_only_sam_many2many.pth")
        model = BaseModel(opt_semsam, build_model_semantic_sam(opt_semsam)).from_pretrained(semsam_ckpt).eval().to(device)
    elif model_name == 'sam':
        # sam_ckpt = "./sam_vit_h_4b8939.pth"
        sam_ckpt = SAM_CKPT
        model = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().to(device)
    # elif model_name == 'seem':
    #     seem_cfg = os.path.join(ROOT_DIR, "configs/seem_focall_unicl_lang_v1.yaml")
    #     seem_ckpt = os.path.join(ROOT_DIR, "seem_focall_v1.pt")
    #     opt_seem = load_opt_from_config_file(seem_cfg)
    #     opt_seem = init_distributed_seem(opt_seem)
    #     model = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().to(device)
    #     with torch.no_grad():
    #         with torch.autocast(device_type='cuda', dtype=torch.float16):
    #             model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    elif model_name == 'gt':
        model = None
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    return (model, model_name)


def get_level_for_semantic_sam(slider=2.0):
    if slider < 1.5 + 0.14:                
        level = [1]
    elif slider < 1.5 + 0.28:
        level = [2]
    elif slider < 1.5 + 0.42:
        level = [3]
    elif slider < 1.5 + 0.56:
        level = [4]
    elif slider < 1.5 + 0.70:
        level = [5]
    elif slider < 1.5 + 0.84:
        level = [6]
    else:
        level = [6, 1, 2, 3, 4, 5]
    return level


@torch.no_grad()
def inference(image_file, model, sam_mode="default", slider=2.0, alpha=0.1, resize_to_original=True, label_mode ="1", anno_mode=['Mask', 'Mark'], gt_ann=None, output_mode="binary_mask"):
    model, model_name = model
    
    with File.open(image_file, "rb") as f:
        image = Image.open(f)

        # if image has only two dimensions (H, W), add one dimension (C)
        if len(image.size) == 2:
            image = image.convert('RGB')
        
        w,h = image.size

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    if model_name == "gt":
        output, mask = som_w_gt(image,  gt_ann, text_size, label_mode, anno_mode)
    else:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            semantic=False

            if model_name == 'semantic-sam':
                level = get_level_for_semantic_sam(slider=slider)
                output, mask = inference_semsam_m2m_auto(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, output_mode=output_mode)
            elif model_name == 'sam':
                output, mask = inference_sam_m2m_auto(model, image, sam_mode, text_size, label_mode, alpha, anno_mode, output_mode=output_mode)
            # elif model_name == 'seem':
            #     output, mask = inference_seem_pano(model, image, text_size, label_mode, alpha, anno_mode, output_mode=output_mode)
    
    if resize_to_original:
        mask = [resize_region(m, w, h) for m in mask]
    return output, mask


def main(args):
    if args.output_folder is not None:
        if args.model_name == "semantic-sam":
            sub_folder = args.model_name + f"_slider_{args.slider}"
        else:
            sub_folder = args.model_name
        if args.anno_mode != ['Mask', 'Mark']:
            sub_folder += "_" + "_".join(args.anno_mode)
        if args.output_mode == "coco_rle":
            sub_folder += f"_rle"
        args.output_folder = os.path.join(args.output_folder, "som", sub_folder)
        os.makedirs(args.output_folder, exist_ok=True)
    
    if args.model_name == "gt":
        assert args.annotation_file is not None and File.isfile(args.annotation_file), "annotation_file is required for gt model"

    gt_ann = None
    if args.annotation_file is None or not File.isfile(args.annotation_file):
        print(f"annotation_file {args.annotation_file} does not exist")
        print(f"walking through image_folder {args.image_folder}")
        print("might be slow......")
        # get all image files under image_folder
        # include all image extensions
        exts = set(['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF', '.gif', '.GIF', '.webp', '.WEBP'])
        image_files = []
        image_ids = []
        for root, dirs, files in os.walk(args.image_folder):
            for file in files:
                id_, ext = os.path.splitext(file)
                if ext in exts:
                    image_files.append(os.path.join(root, file))
                    image_ids.append(id_)
    else:
        print("annotation_file is provided, loading image_files from annotation_file")
        if args.annotation_file.endswith('image_ids.json') or "/ids/" in args.annotation_file:
            image_ids = json.load(File.open(args.annotation_file))
            # check if image_ids are with extensions already
            image_files = [
                os.path.join(args.image_folder, f"{id_}.jpg")
                if not str(id_).endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".bmp", ".BMP", ".tif", ".tiff", ".TIF", ".TIFF", ".gif", ".GIF", ".webp", ".WEBP"))
                else os.path.join(args.image_folder, f"{id_}")
                for id_ in image_ids]
        elif args.annotation_file.endswith('.json'):
            annotations = json.load(File.open(args.annotation_file))
            image_files = []
            if "aokvqa" in args.annotation_file:
                split2folder = {"train": "train2014", "val": "val2014", "test": "test2015"}
                image_ids = []
                for d in annotations:
                    sub_folder = split2folder[d["split"]]
                    image_id = d["image_id"]
                    image_file = os.path.join(args.image_folder, sub_folder, f"COCO_{sub_folder}_{image_id:012d}.jpg")
                    image_files.append(image_file)
                    image_ids.append(f"COCO_{sub_folder}_{image_id:012d}")
            elif "gqa" in args.annotation_file:
                image_ids = set([d['imageId'] for d in annotations.values()])
                for id_ in image_ids:
                    image_file = os.path.join(args.image_folder, f"{id_}.jpg")
                    image_files.append(image_file)
            else:
                image_files = [os.path.join(args.image_folder, annotation['image_name']) for annotation in annotations]
                image_ids = [annotation['image_name'] for annotation in annotations]
        elif args.annotation_file.endswith('.jsonl'):
            annotations = [json.loads(line) for line in File.open(args.annotation_file)]
            if "vcr" in args.annotation_file:
                image_files = [os.path.join(args.image_folder, annotation['img_fn']) for annotation in annotations]
                image_ids = [annotation['img_fn'] for annotation in annotations]
            elif "gqa/val_balanced_gqa" in args.annotation_file or "gqa/train_balanced_gqa" in args.annotation_file:
                image_files = [os.path.join(args.image_folder, f"{annotation['vg_id']}.jpg") for annotation in annotations]
                gt_ann =  {os.path.join(args.image_folder, f"{annotation['vg_id']}.jpg"): annotation["scene_graph"] for annotation in annotations}
                image_ids = [annotation['vg_id'] for annotation in annotations]
            elif "gqa/val_cot":
                image_files = [os.path.join(args.image_folder, f"{annotation['image_id']}.jpg") for annotation in annotations]
                image_ids = [annotation['image_id'] for annotation in annotations]
            elif "vsr" in args.annotation_file:
                image_files = []
                images = set([d['image_link'] for d in annotations])
                image_ids = []
                for im_ in images:
                    im_ = "/".join(im_.split("/")[-2:])
                    image_file = os.path.join(args.image_folder, im_)
                    image_files.append(image_file)
                    image_ids.append(im_)
            else:
                image_files = [os.path.join(args.image_folder, annotation['image_name']) for annotation in annotations]
                image_ids = [annotation['image_name'] for annotation in annotations]
        image_files = list(set(image_files))
        # if args.debug:
        #     image_files = sorted(image_files)[:100]
        # image_files = sorted(image_files)
        # sort  image ids and image files according to image ids
        image_files = [image_file for _, image_file in sorted(zip(image_ids, image_files))]
        image_ids = [image_id for image_id, _ in sorted(zip(image_ids, image_files))]
        if args.debug:
            image_files = image_files[:100]
            image_ids = image_ids[:100]

    model = build_som_model(args.model_name)
    for image_id, image_file in tqdm(zip(image_ids, image_files), total=len(image_files)):
        image_file_name, _ = os.path.splitext(image_id)

        output_image_file = os.path.join(args.output_folder, f"{image_file_name}.jpg")
        output_mask_file = os.path.join(args.output_folder, f"{image_file_name}.pkl")
        os.makedirs(os.path.dirname(output_mask_file), exist_ok=True)
        parent_folder = os.path.dirname(output_image_file)
        # os.makedirs(parent_folder, exist_ok=True)
        compress_dir = os.path.dirname(output_mask_file)+"_rle"
        # os.makedirs(compress_dir, exist_ok=True)
        if not args.debug and not args.overwrite and File.isfile(output_mask_file):
            print(f"output_mask_file {output_mask_file} already exist, skipping this image")
            if args.compress and not args.output_folder.endswith("rle"):
                compress_mask_file = os.path.join(compress_dir, f"{image_file_name}.pkl")
                compress_seg(image_file, output_mask_file, compress_mask_file)
            continue
        if "movieclips_Waiting..." in image_file:
            image_file = image_file.replace("movieclips_Waiting...", "movieclips_Waiting")
        elif "movieclips_48_Hrs." in image_file:
            image_file = image_file.replace("movieclips_48_Hrs.", "movieclips_48_Hrs")
        if gt_ann is not None:
            ann = gt_ann[image_file]
        else:
            ann = None
        
        if args.debug:
            output, mask = inference(image_file, model, slider=args.slider, anno_mode=args.anno_mode,
                                    gt_ann=ann, output_mode=args.output_mode)
            if mask is None:
                print(f"Failed to process image {image_file}")
                continue
            if output is not None:
                # output is a numpy array
                # save output as a jpg file
                output = Image.fromarray(output)
                with File.open(output_image_file, "wb") as f:
                    output.save(f)

            # save mask, list of dictionary
            # save mask as a json file
            with File.open(output_mask_file, 'wb') as f:
                pickle.dump(mask, f)
        else:
            try:
                output, mask = inference(image_file, model, slider=args.slider, anno_mode=args.anno_mode,
                                        gt_ann=ann, output_mode=args.output_mode)
            except Exception as e:
                print("Error processing image", image_file)
                print(e)
                print("skipping this image for now")
                continue
            else:
                if mask is None:
                    print(f"Failed to process image {image_file}")
                    continue
                if output is not None:
                    # output is a numpy array
                    # save output as a jpg file
                    output = Image.fromarray(output)
                    with File.open(output_image_file, "wb") as f:
                        output.save(f)

                # save mask, list of dictionary
                # save mask as a json file
                with File.open(output_mask_file, 'wb') as f:
                    pickle.dump(mask, f)
                if args.compress and args.output_mode == "binary_mask":
                    compress_mask_file = os.path.join(compress_dir, f"{image_file_name}.pkl")
                    compress_seg(image_file, output_mask_file, compress_mask_file)
                
        print(output_mask_file)
        
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default='../data/coco')
    parser.add_argument('--model_name', type=str, default='semantic-sam')
    parser.add_argument('--output_mode', type=str, default='binary_mask', choices=['binary_mask', 'coco_rle'])
    parser.add_argument('--anno_mode', type=str, nargs="+", default=["Mask", "Mark"])

    # sam params
    parser.add_argument('--sam_mode', choices=['whole', 'default', 'part'], help='mode to run', default='whole')

    # sem-sam params
    parser.add_argument('--slider', type=float, default=2.0)

    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--overwrite', action="store_true", default=False)
    parser.add_argument('--compress', action="store_true", default=False)
    parser.add_argument('--resize', action="store_true", default=False, help='resize mask to original image size.')
    args = parser.parse_args()
    main(args)
