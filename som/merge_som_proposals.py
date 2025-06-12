from glob import glob
from torchvision.ops import nms
import pickle
import json
from tqdm import tqdm
from multiprocessing import Pool
# import shutil
from azfuse import File
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Literal
import torch
from PIL import Image
from pycocotools import mask as maskUtils

from skimage.transform import resize 

def load_all_gqa_ids(debug=False):
    gqa_ids = set()
    # data/gqa/val_balanced_gqa_GQA_test_200_coco_captions_region_captions_scene_graphs_aokvqa.jsonl
    val_small = "linjli/data/raw_data/gqa/val_balanced_gqa_GQA_test_200_coco_captions_region_captions_scene_graphs_aokvqa.jsonl"
    with File.open(val_small, "r") as f:
        for line in f.readlines():
            line = line.strip()
            gqa_data = json.loads(line)
            gqa_ids.add(str(gqa_data["vg_id"]))
    if debug:
        print(f"Loaded {len(list(gqa_ids))} gqa_ids from val_small")
        return sorted(list(gqa_ids))

    aokvqa = "linjli/data/raw_data/gqa/aokvqa_gqa_image_ids.json"
    with File.open(aokvqa, "r") as f:
        aokvqa_data = json.load(f)
        for gqa_data in aokvqa_data:
            gqa_ids.add(str(gqa_data))

    train = "linjli/data/raw_data/gqa/train_balanced_gqa_coco_captions_region_captions_scene_graphs_aokvqa.jsonl"
    with File.open(train, "r") as f:
        for line in f.readlines():
            line = line.strip()
            gqa_data = json.loads(line)
            gqa_ids.add(str(gqa_data["vg_id"]))
    
    val = "linjli/data/raw_data/gqa/val_balanced_gqa_coco_captions_region_captions_scene_graphs.jsonl"
    with File.open(val, "r") as f:
        for line in f.readlines():
            line = line.strip()
            gqa_data = json.loads(line)
            gqa_ids.add(str(gqa_data["vg_id"]))

    print(f"Loaded {len(list(gqa_ids))} gqa_ids")
    return sorted(list(gqa_ids))

def load_som_annotation(file_):
    # Load the annotations from the SoM model
    som_annotation = pickle.load(File.open(file_, "rb"))
    # som_annotation is a list
    return som_annotation

# nms(boxes: Tensor, scores: Tensor, iou_threshold: float)
def load_different_som_annotations(input_files):
    som_annotations = []
    for file_ in input_files:
        som_annotation = load_som_annotation(file_)
        som_annotations.extend(som_annotation)
    return som_annotations

def merge_som_annotations(som_annotations, iou_threshold=0.6, sort_by: Literal['area', 'score']='area', merge_mode: Literal['nms', 'nmm']='nms'):
    to_merge_boxes = []
    scores = []
    for i, som_annotation in enumerate(som_annotations):
        boxes = som_annotation["bbox"] # x1, y1, w, h
        transformed_boxes = torch.Tensor(boxes).view(-1, 4)
        transformed_boxes[:, 2] = transformed_boxes[:, 0] + transformed_boxes[:, 2]
        transformed_boxes[:, 3] = transformed_boxes[:, 1] + transformed_boxes[:, 3]
        to_merge_boxes.append(transformed_boxes)

        if sort_by == 'score':
            scores.append(som_annotation["stability_score"])
        else:
            scores.append(som_annotation["area"])
    scores = torch.Tensor(scores)
    to_merge_boxes = torch.cat(to_merge_boxes, dim=0)
    # Apply NMS
    print(f"Applying NMS with {len(to_merge_boxes)} boxes")
    indices = nms(to_merge_boxes[:, :4], scores, iou_threshold)
    indices = indices.tolist()
    print(f"Got {len(indices)} boxes after NMS")
    return [som_annotations[i] for i in indices]


def process_image(image_id, output_folder, som_parent_folder: str, iou_threshold: float, som_model: str, modes: list[str]=None, overwrite: bool=False):
    output_file = f"{output_folder}/{image_id}.pkl"
    if File.isfile(output_file) and not overwrite:
        print(f"Skipped {output_file}, already exists.")
        return 1
    missing = 0
    input_files = []

    # modes = {
    #     'sam': ['part', 'whole'],
    #     'semantic-sam': ['slider_1.5', 'slider_1.7', 'slider_2.0']
    # }

    if not modes:
        input_file_rle = f"{som_parent_folder}/{som_model}_rle/{image_id}.pkl"
        input_file = f"{som_parent_folder}/{som_model}/{image_id}.pkl"
        if not File.isfile(input_file) and not File.isfile(input_file_rle):
            missing += 1
            print(f"Missing {input_file} and {input_file_rle}")
        elif File.isfile(input_file_rle):
            input_files.append(input_file_rle)
        else:
            input_files.append(input_file)
    else:
        for mode in modes:
            input_file_rle = f"{som_parent_folder}/{som_model}_{mode}_rle/{image_id}.pkl"
            input_file = f"{som_parent_folder}/{som_model}_{mode}/{image_id}.pkl"
            if not File.isfile(input_file) and not File.isfile(input_file_rle):
                missing += 1
                print(f"Missing {input_file} and {input_file_rle}")
            elif File.isfile(input_file_rle):
                input_files.append(input_file_rle)
            else:
                input_files.append(input_file)
            
    som_annotations = load_different_som_annotations(input_files)
    merged = merge_som_annotations(som_annotations, iou_threshold)
    with File.open(output_file, "wb") as f:
        pickle.dump(merged, f)
    print(f"Saved to {output_file}, with {len(merged)} boxes")
    return 1
    
def merge_som_annotations_for_gqa(som_parent_folder, som_model=["seem", "semantic-sam"], sliders=[1.5, 1.7, 2.0], iou_threshold=0.7, overwrite=False):
    # gqa_ids = load_all_gqa_ids()
    gqa_ids = load_all_gqa_ids(debug=True)
    som_setting = "_".join(som_model)
    output_folder = f"{som_parent_folder}/merged/{som_setting}_slider_{'-'.join([str(d) for d in sliders])}_nms{iou_threshold}"
    # os.makedirs(output_folder, exist_ok=True)
    # to_prepare = []
    # for slider in sliders:
    #     slider_sub_folder = f"{som_parent_folder}/{som_model}_slider_{slider}"
    #     input_files = [f"{slider_sub_folder}/{image_id}.pkl" for image_id in gqa_ids]
    #     to_prepare.extend(input_files)
    # File.prepare(to_prepare)
    # print(f"Preparing {len(to_prepare)} files")

    # make the following multi-threaded
    from functools import partial
    process_image_partial = partial(process_image, output_folder=output_folder, som_parent_folder=som_parent_folder, iou_threshold=iou_threshold, 
                                    som_model=som_model, sliders=sliders, overwrite=overwrite)
    progress_bar = tqdm(total=len(gqa_ids))
    # Create a pool of processes
    with Pool(processes=16) as pool:
        # Use map to process images in parallel
        # results = list(tqdm(pool.imap(process_image_partial, gqa_ids), total=len(gqa_ids)))
        # results = pool.starmap(process_image_partial, gqa_ids)
        for result in pool.imap(process_image_partial, gqa_ids, chunksize=16):
            progress_bar.update(result)

    # for image_id in tqdm(gqa_ids):
    #     output_file = f"{output_folder}/{image_id}.pkl"
    #     if os.path.exists(output_file) and not overwrite:
    #         continue
    #     som_annotations = load_different_som_annotations(som_parent_folder, image_id, som_model, sliders)
    #     merged = merge_som_annotations(som_annotations, iou_threshold)
    #     print(f"Saving to {output_file}, with {len(merged)} boxes")
    #     pickle.dump(merged, open(output_file, "wb"))


def merge_som_annotations_for_vsr(som_parent_folder, som_model=["semantic-sam"], sliders=[1.5, 1.7, 2.0], iou_threshold=0.7, overwrite=False):
    annotations = [json.loads(d) for d in File.open("../data/vsr/test.jsonl", "r").readlines()]
    print(f"Loaded {len(annotations)} annotations")
    images = set([d['image_link'].replace(".jpg", "") for d in annotations])
    image_ids = []
    for im_ in images:
        im_ = os.path.splitext(im_.split("/")[-1])[0]
        image_ids.append(im_)
    som_setting = "-".join(som_model)
    output_folder = f"{som_parent_folder}/merged/{som_setting}_slider_{'-'.join([str(d) for d in sliders])}_nms{iou_threshold}"
    # os.makedirs(output_folder, exist_ok=True)
    # to_prepare = []
    # for slider in sliders:
    #     slider_sub_folder = f"{som_parent_folder}/{som_model}_slider_{slider}"
    #     input_files = [f"{slider_sub_folder}/{image_id}.pkl" for image_id in gqa_ids]
    #     to_prepare.extend(input_files)
    # File.prepare(to_prepare)
    # print(f"Preparing {len(to_prepare)} files")

    # make the following multi-threaded
    from functools import partial
    process_image_partial = partial(process_image, output_folder=output_folder, som_parent_folder=som_parent_folder, iou_threshold=iou_threshold, 
                                    som_model=som_model, sliders=sliders, overwrite=overwrite)
    progress_bar = tqdm(total=len(image_ids))
    # Create a pool of processes
    with Pool(processes=16) as pool:
        for result in pool.imap(process_image_partial, image_ids, chunksize=16):
            progress_bar.update(result)

def merge_som_annotations_for_all_videos(som_parent_folder, som_model="sam", modes=["part", "whole"], iou_threshold=0.6, overwrite=False):

    # som_parent_folder = som_sam_detections
    #   som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_part_rle/0001.pkl
    #   som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_whole_rle/0001.pkl
    video_ids = os.listdir(som_parent_folder)
    for video_id in video_ids:
        video_path = f"{som_parent_folder}/{video_id}/som"
        if not os.path.isdir(video_path):
            print(f"Skipping {video_path}")
            continue
        merge_som_annotations_for_video(video_path, som_model=som_model, modes=modes, iou_threshold=iou_threshold, overwrite=overwrite)
        break

def merge_som_annotations_for_video(video_path, som_model="sam", modes=["part", "whole"], iou_threshold=0.6, overwrite=False):

    # video_path = som_sam_detections/{video_id}/som
    #   som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_part_rle/0001.pkl
    #   som_sam_detections/03f2ed96-1719-427d-acf4-8bf504f1d66d/som/sam_whole_rle/0001.pkl
    
    output_folder = f"{video_path}/{som_model}_{'-'.join([str(d) for d in modes])}_nms{iou_threshold}"

    # Get frame image_id from common video_path
    frame_ids = sorted(glob(f"{video_path}/*/*.pkl"))
    frame_ids = [os.path.splitext(os.path.basename(image_id))[0] for image_id in frame_ids]
    from functools import partial
    process_image_partial = partial(process_image, output_folder=output_folder, som_parent_folder=video_path,
                                    som_model=som_model, modes=modes, iou_threshold=iou_threshold, overwrite=overwrite)
    progress_bar = tqdm(total=len(frame_ids))
    # Create a pool of processes
    num_processes = 16
    with Pool(processes=num_processes) as pool:
        for result in pool.imap(process_image_partial, frame_ids, chunksize=num_processes):
            progress_bar.update(result)

def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def transform_gt_bb_into_som(gt_bb, image_size):
    # short_edge will be resize to 640
    width, height = image_size
    short_edge = min(width, height)
    scale = 640 / short_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    x1, y1, x2, y2 = gt_bb
    x1 = int(x1 * scale)
    y1 = int(y1 * scale)
    x2 = int(x2 * scale)
    y2 = int(y2 * scale)
    return [x1, y1, x2, y2]


def calculate_iou_with_references(som_folder, annotation_file="../data/gqa/val_cot_gqa_relevant_regions_sam_hq.jsonl", iou_threshold=0.5):
    annotations = [json.loads(line) for line in File.open(annotation_file, "r").readlines()]
    ious = []
    valid_regions = 0
    total_candidates = 0
    for ann in tqdm(annotations):
        image_id = ann["image_id"]
        seg = load_som_annotation(f"{som_folder}/{image_id}.pkl")
        regions = ann["relevant_regions"]
        total_candidates += len(seg)
        for region in regions:
            gt_bb = region["bbox"]
            gt_bb = transform_gt_bb_into_som(gt_bb, (ann["width"], ann["height"]))
            max_iou = 0
            for item in seg:
                bb = item["bbox"]
                x, y, w, h = bb
                bb = [x, y, x+w, y+h]
                iou_score = iou(gt_bb, bb)
                max_iou = max(max_iou, iou_score)
            if max_iou >= iou_threshold:
                valid_regions += 1
            ious.append(max_iou)
    print(f"Average iou: {sum(ious)/len(ious)}")
    print(f"Valid regions: {valid_regions} / {len(ious)}")
    print(f"Total candidates: {total_candidates}")


def compress_seg(image_file, som_file, output_file):
    from PIL import Image
    import numpy as np
    if not File.isfile(image_file):
        print(f"Image {image_file} does not exist")
        return 1
    if not File.isfile(som_file):
        print(f"File {som_file} does not exist")
        return 1
    if File.isfile(output_file):
        print(f"Skipped {image_file}, already exists.")
        return 1
    p = pickle.load(File.open(som_file,'rb')) # Size: 4.8M
    # compress segmentation
    with File.open(image_file, "rb") as f:
        w,h = Image.open(f).size
    segmentations = [d['segmentation'] for d in p]
    uncompressed_rle = mask_to_rle_pytorch(torch.BoolTensor(segmentations))
    rles = [coco_encode_rle(r) for r in uncompressed_rle ]

    # sanity check
    mask = [annToMask(rle, h, w) for rle in rles] # np.uint8 array
    for i in range(len(p)):
        assert np.sum(p[i]['segmentation'] - mask[i]) == 0, i

    # reassign segmentation
    for i in range(len(p)):
        p[i]['segmentation'] =  rles[i]

    # os.makedirs(f'..data/som/gqa/som/semantic-sam_slider_1.5_rle', exist_ok=True)
    # Save output
    with File.open(output_file,'wb') as f:
        pickle.dump(p, f)
    return 1


def compress_seg_for_gqa(image_folder, som_folder):
    gqa_ids = load_all_gqa_ids()
    image_files = [image_folder + "/" + gqa_id + ".jpg" for gqa_id in gqa_ids]
    som_files = [f"{som_folder}/{gqa_id}.pkl" for gqa_id in gqa_ids]

    output_folder = f"{som_folder}_rle"
    output_files = [f"{output_folder}/{gqa_id}.pkl" for gqa_id in gqa_ids]
    progress_bar = tqdm(total=len(gqa_ids))
    with Pool(processes=16) as pool:
        for result in pool.starmap(compress_seg, zip(image_files, som_files, output_files)):
            progress_bar.update(result)


def compress_seg_for_vsr(image_folder, som_folder):
    annotations = [json.loads(d) for d in File.open("../data/vsr/test.jsonl", "r").readlines()]
    print(f"Loaded {len(annotations)} annotations")
    images = set([d['image_link'].replace(".jpg", "") for d in annotations])
    im_ids = []
    for im_ in images:
        im_ = "/".join(im_.split("/")[-2:])
        im_ids.append(im_)
    image_files = [f"{image_folder}/{im}.jpg" for im in im_ids]
    som_files = [f"{som_folder}/{os.path.basename(im)}".replace(".jpg", ".pkl") for im in image_files]

    output_folder = f"{som_folder}_rle"
    output_files = [f"{output_folder}/{os.path.basename(im)}".replace(".jpg", ".pkl") for im in image_files]
    progress_bar = tqdm(total=len(im_ids))
    with Pool(processes=16) as pool:
        for result in pool.starmap(compress_seg, zip(image_files, som_files, output_files)):
            progress_bar.update(result)


## Functions from segment-anything
## https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/amg.py#L107
def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/amg.py#L294C1-L300C15
def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle
#######


# Code from Osprey to decode segmentation mask as np.uint8 array
def annToMask(mask_ann: List[Dict], h, w) -> np.ndarray:
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def resize_region(region: Dict, new_w: int, new_h: int):
    """
    Resize SAM segmentation to new width and height.

    Parameters:
    region (dict): The region to resize. It should contain 'segmentation' key with 'size' subkey.
    new_w (int): The new width for the region.
    new_h (int): The new height for the region.

    Returns:
    dict: The resized region. If the original size is the same as the new size, the original region is returned.
    """
    # Retrieve original size
    h, w = region['segmentation']['size']
    if new_h == h and new_w == w:
        return region
    
    if isinstance(region['segmentation'], np.ndarray):
        mask = region['segmentation']
    else:
        mask: np.ndarray = annToMask(region['segmentation'], h, w)
    
    # Calculate resize ratios
    w_ratio = new_w / w
    h_ratio = new_h / h
    
    # Resize mask
    resized_mask = resize(mask, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

    # Convert resized mask back to RLE for efficiency
    resized_rle = maskUtils.encode(np.asfortranarray(resized_mask))
    
    # Update bbox and xyxy with the new ratios
    bbox = region['bbox'] # [xywh]
    resized_bbox = [bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio]
    
    if 'xyxy' not in region:
        x,y,xw,yh = bbox
        xyxy = [x, y, x+xw, y+yh]
        region['xyxy'] = xyxy
    xyxy = region['xyxy']
    resized_xyxy = [xyxy[0] * w_ratio, xyxy[1] * h_ratio, xyxy[2] * w_ratio, xyxy[3] * h_ratio]

    norm_xyxy = [int(xyxy[0] / w * 1000), int(xyxy[1] / h * 1000), int(xyxy[2] / w * 1000), int(xyxy[3] / h * 1000)]
    for c in norm_xyxy:
        assert 0 <= c <= 1000
    
    # Update area based on the mask
    resized_area = np.sum(resized_mask)
    
    # Return updated region dictionary
    return {
        'segmentation': resized_rle,
        'area': resized_area,
        'bbox': resized_bbox,
        'predicted_iou': region['predicted_iou'],  # Presuming IOU remains the same, might not be true in all cases
        'point_coords': region['point_coords'],
        'stability_score': region['stability_score'],  # Assuming stability score remains the same
        'crop_box': [0, 0, new_w, new_h],
        'xyxy': resized_xyxy,
        'norm_xyxy': norm_xyxy,
    }


if __name__ == "__main__":
    # som_parent_folder = "../data/gqa/som"
    # merge_som_annotations_for_image(som_parent_folder)
    # from fire import Fire
    # Fire()

    som_parent_folder = "som_sam_detections"
    merge_som_annotations_for_all_videos(som_parent_folder, som_model="sam", modes=["part", "whole"], iou_threshold=0.6, overwrite=False)
    mode_dict = {
        "sam": ["part", "whole"],
        "semantic-sam": [1.5, 1.7, 2.0],
    }
