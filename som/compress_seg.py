import pickle
import numpy as np
from typing import List, Dict, Any
import torch
from PIL import Image
from pycocotools import mask as maskUtils

id = 2333703
im = f'../data/vg/{id}.jpg'
p = pickle.load(open(f'..data/som/gqa/som/semantic-sam_slider_1.5/{id}.pkl','rb')) # Size: 4.8M

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

# compress segmentation
w,h = Image.open(im).size
segmentations = [d['segmentation'] for d in p]
uncompressed_rle = mask_to_rle_pytorch(torch.BoolTensor(segmentations))
rles = [coco_encode_rle(r) for r in uncompressed_rle ]

# sanity check
mask: List[np.ndarray] = [annToMask(rle, h, w) for rle in rles] # np.uint8 array
for i in range(len(p)):
    assert np.sum(p[i]['segmentation'] - mask[i]) == 0, i

# reassign segmentation
for i in range(len(p)):
    p[i]['segmentation'] =  rles[i]

os.makedirs(f'..data/som/gqa/som/semantic-sam_slider_1.5_rle', exist_ok=True)
# Save output
pickle.dump(p, open(f'..data/som/gqa/som/semantic-sam_slider_1.5_rle/{id}_rle.pkl','wb')) # Size: 128K
