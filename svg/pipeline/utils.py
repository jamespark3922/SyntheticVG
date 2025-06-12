from typing import Any, List, Dict
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
from pycocotools import mask as maskUtils

def create_region_annotations(masks: np.ndarray, boxes: list | np.ndarray=None, output_mode='rle') -> list[dict]:
    """Convert binary masks and bounding boxes to region annotations.

    Args:
        masks (np.ndarray): Binary segmentation masks of shape (N, H, W)
        boxes (list | np.ndarray, optional): Bounding boxes in XYXY format.
            If None, boxes will be computed from masks. Defaults to None.

    Returns:
        list[dict]: List of region annotations, each containing:
            - segmentation: RLE encoded mask
            - bbox: [x, y, width, height] format
            - area: number of pixels in mask
    
    Raises:
        AssertionError: If number of boxes doesn't match number of masks
    """
    areas = masks.sum(axis=(1, 2))
    if boxes is None:
        boxes = [mask_to_box(mask) for mask in masks]
    boxes = xyxy_to_xywh(boxes).tolist() # xywh boxes

    assert len(boxes) == len(masks)
    if output_mode == 'rle':
        masks = numpy_mask_to_rle(masks)
    regions = [{'segmentation': m} for m in masks]
    for area, box, region in zip(areas, boxes, regions):
        region['bbox'] = box
        region['area'] = float(area)
    
    assert len(regions) == len(masks)
    return regions

#######
# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/amg.py
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

def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:

    h, w = uncompressed_rle["size"]
    rle = maskUtils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle
#######

def numpy_mask_to_rle(mask: np.ndarray) -> list[dict]:
    uncompressed_rle = mask_to_rle_pytorch(torch.BoolTensor(mask))
    rles = [coco_encode_rle(r) for r in uncompressed_rle ]
    return rles

def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    return np.array([[
        box[0], box[1], box[2] - box[0], box[3] - box[1]
    ] for box in boxes])

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    return np.array([[
        box[0], box[1], box[0] + box[2], box[1] + box[3]
    ] for box in boxes])

def mask_to_box(mask: np.ndarray | torch.Tensor) -> list[float]:
    """Convert binary mask to bounding box in XYXY format."""
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)
    if mask.dim() == 2:
        mask.unsqueeze_(0)
    
    assert mask.dim() == 3, f"mask dim should be 3, got {mask.dim()}"

    if mask.sum() == 0:
        print("mask sum is 0")
        return [0, 0, 0, 0]
    bbox = masks_to_boxes(mask)[0].numpy() # [x1, y1, x2, y2]
    return bbox

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, np.ndarray):
        return mask_ann
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list): # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else: # rle
        rle = mask_ann
    mask = maskUtils.decode(rle).astype(bool)
    return mask