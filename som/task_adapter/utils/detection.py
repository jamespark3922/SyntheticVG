'''
Helper utils for handling detections and tracking based on supervision libary.
'''

import pickle
from typing import Dict, List, Literal, Union
import cv2
import numpy as np

from tqdm import tqdm

from pycocotools import mask as maskUtils
import supervision as sv

def load_image(image: str | np.ndarray) -> np.ndarray:
    """
    Loads BGR image from file path or numpy array.
    Note: supervision library uses BGR format when loading and displaying images.
    """
    if isinstance(image, np.ndarray):
        return image
    else:
        return cv2.imread(image)

def annToMask(mask_ann, h=None, w=None) -> np.ndarray:
    """
    Decodes masks annotation to numpy array
    """
    if isinstance(mask_ann, np.ndarray):
        return mask_ann

    if isinstance(mask_ann, list): # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list): # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else: # compressed RLE
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def annotate(image: str | np.ndarray, detections: sv.Detections, labels=None, mode='mask', plot=True):
    """
    Annotates the given image with the provided detections.

    Args:
        image (str or np.ndarray): The path to the image file or the image array.
        detections (sv.Detections): The detections to annotate on the image.
        labels (list, optional): The labels corresponding to the detections. Defaults to None.
        mode (str, optional): The annotation mode. Can be 'mask' or 'box'. Defaults to 'mask'.
        plot (bool, optional): Whether to plot the annotated image. Defaults to True.

    Returns:
        np.ndarray: The annotated image.
    """
    
    image: np.ndarray = load_image(image)
        
    box_annotator = sv.BoundingBoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

    if mode == 'mask':
        annotated_image = mask_annotator.annotate(
            scene=image, detections=detections)
    else:
        annotated_image = box_annotator.annotate(
            scene=image, detections=detections)
    
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    if plot:
        sv.plot_image(annotated_image)

    return annotated_image

def merge_detections(
        detections_list: list[sv.Detections], 
        iou_threshold: float = 0.5,
        mode: Literal['nms', 'nmm'] = 'nms'
    ) -> sv.Detections:
    """
    Merge multiple detections into a single set of detections.

    Args:
        detections_list (List[sv.Detections]): A list of detections to be merged.
        iou_threshold (float, optional): The IoU threshold for merging overlapping detections. Defaults to 0.5.
        mode (Literal['nms', 'nmm'], optional): The merge mode to be used.  
            'nms' for non-maximum suppression, 'nmm' for non-maximum merging. Defaults to 'nms'.

    Returns:
        sv.Detections: The merged set of detections.

    Raises:
        ValueError: If an invalid merge mode is provided.

    """
    detections_merged = sv.Detections.merge(detections_list)

    if mode == 'nms':
        detections = detections_merged.with_nms(threshold=iou_threshold, class_agnostic=True)
    elif mode == 'nmm':
        detections = detections_merged.with_nmm(threshold=iou_threshold, class_agnostic=True)
    else:
        raise ValueError(f"Invalid merge mode: {mode}")
    
    return detections

def get_detections_from_sam(sam_result: list[dict]) -> sv.Detections:
    """
    Creates a Detections instance from
    [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
    inference result.

    Args:
        sam_result (List[dict]): The output Results instance from SAM

    Returns:
        Detections: A new Detections object.

    Example:
        ```python
        import supervision as sv
        from segment_anything import (
            sam_model_registry,
            SamAutomaticMaskGenerator
         )

        sam_model_reg = sam_model_registry[MODEL_TYPE]
        sam = sam_model_reg(checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        sam_result = mask_generator.generate(IMAGE)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        ```
    """

    sorted_generated_masks = sorted(
        sam_result, key=lambda x: x["area"], reverse=True
    )

    xywh = np.array([mask["bbox"] for mask in sorted_generated_masks])
    mask = np.array([annToMask(mask["segmentation"]) for mask in sorted_generated_masks], dtype=bool)

    if np.asarray(xywh).shape[0] == 0:
        return sv.Detections.empty()

    xyxy = sv.detection.utils.xywh_to_xyxy(boxes_xywh=xywh)
    detections = sv.Detections(xyxy=xyxy, mask=mask)
    detections.class_id = np.arange(len(detections))
    detections.confidence = np.array([r['stability_score'] for r in sorted_generated_masks])

    return detections

def get_detections_from_pickle(pickle_path: str) -> sv.Detections:
    """
    Load SAM detections from a pickle file.

    Args:
        pickle_path (str): The path to the pickle file.

    Returns:
        Detections: A new Detections object.
    """
    with open(pickle_path, 'rb') as f:
        sam_detections: list[dict] = pickle.load(f)
    
    detections = get_detections_from_sam(sam_detections)
    return detections

 
def track_from_detections(
    images: List, detections_list: List[sv.Detections], 
    video_info: sv.VideoInfo, output_path, 
    draw_box=True,
    draw_mask=False,
    label_position: sv.Position = 'CENTER_OF_MASS',
    disable_tqdm=True
):
    """
    Process a list of images with corresponding detections and generate a video with annotated frames.

    Args:
        images (List): A list of image paths or image objects.
        detections_list (List[sv.Detections]): A list of detection objects.
        width (int): The width of the output video frames.
        height (int): The height of the output video frames.
        fps (int): The frame rate of the output video.
        output_path: The path to save the output video.
        draw_box (bool, optional): Whether to draw bounding boxes around the detections. Defaults to True.
        draw_mask (bool, optional): Whether to draw masks around the detections. Defaults to False.
        label_position (sv.Position, optional): The position of the labels. Defaults to 'CENTER_OF_MASS'.
        disable_tqdm (bool, optional): Whether to disable the progress bar. Defaults to True.

    Returns:
        Saves tracked video into output_path
    """
    
    byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(label_position)
    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        
        # Iterate through images.
        for image, detections in tqdm(zip(images, detections_list), disable=disable_tqdm):
                
            # Update detections with tracker ids fro byte_tracker.
            tracked_detections = byte_tracker.update_with_detections(detections)
    
            # Load frame
            frame: np.ndarray = load_image(image)
            
            # Create labels with tracker_id for label annotator.
            labels = [ f"{tracker_id}" for tracker_id in tracked_detections.tracker_id ]
    
            # Apply label annotator to frame.
            annotated_frame = frame.copy()
            if draw_box:
                annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            if draw_mask:
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
    
            # Save the annotated frame to an output video.
            sink.write_frame(frame=annotated_frame)
    
    return



if __name__ == '__main__':
    import pandas as pd
    tqdm.pandas()

    video_id = '03f2ed96-1719-427d-acf4-8bf504f1d66d'
    video_path = f'/net/nfs.cirrascale/mosaic/jamesp/data/PVSG_dataset/ego4d/videos/{video_id}.mp4'
    video_info = sv.VideoInfo.from_video_path(video_path)

    # Load preprocessed detections
    whole_regions_df = pd.read_json(f'/data/jamesp/som/PVSG/ego4d/frames/{video_id}/sam_whole/regions.jsonl', lines=True)
    part_regions_df = pd.read_json(f'/data/jamesp/som/PVSG/ego4d/frames/{video_id}/sam_part/regions.jsonl', lines=True)

    # Merge Dataframe into detections
    regions_df = whole_regions_df.merge(part_regions_df[['image_id', 'regions']], on='image_id', suffixes=['_sam_whole', '_sam_part'], how='left')
    regions_df.head()


    # Load images and detections
    images_list = regions_df['image_path']

    for region_key in ['whole', 'part']:
        detections_list = regions_df['regions'].progress_apply(get_detections_from_sam)

        track_from_detections(  
            images=images_list,
            detections_list=detections_list,
            video_info=video_info,
            output_path=f"{region_key}_output.mp4"
        )
    