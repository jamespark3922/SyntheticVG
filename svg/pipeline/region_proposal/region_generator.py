import numpy as np
from PIL import Image
from enum import IntEnum, Enum

from svg.draw_utils import annToMask, detections_from_sam

from .sam import SamMaskGenerator
from ..grounding.grounding_dino import GroundingDinoSAM
from ..captioning import ObjectCaptioner

class RegionMode(Enum):
    MERGED = 'merged'

    # SAM region modes
    SAM_DEFAULT = 'sam_default'
    SAM_PART = 'sam_part'
    SAM_WHOLE = 'sam_whole'

    # GroundingDINO region modes
    GROUNDING_DINO = 'grounding_dino'

class MergeMode(IntEnum):
    NON_OVERLAP_OR_AREA = 0
    NON_OVERLAP_ONLY = 1
    AREA_ONLY = 2
    NON_OVERLAP_AND_AREA = 3
    OVERLAP_ONLY = 4
    NOT_SAME = 5


def get_boxes(regions) -> np.ndarray:
    return np.array([region['bbox'] for region in regions])

def get_masks(regions) -> np.ndarray:
    return np.array([annToMask(region['segmentation']) for region in regions])

def get_intersection_box(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Given two sets of bounding boxes, computes their intersection areas using numpy.

    Args:
        boxes1: np array (N, 4).
        boxes2: np array (M, 4).
    Returns:
        intersection: np array (N, M).
    """
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = rb - lt
    wh = np.clip(wh, a_min=0, a_max=None)  # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]        # [N,M]
    return inter

def get_intersection_mask(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    """
    Args:
        masks1: np bool array  [N x H x W]
        masks2: np bool array [M x H x W]
    Returns:
        intersection: [N x M]
    """
    return np.einsum('nij,mij->nm', masks1.astype(np.uint8), masks2.astype(np.uint8), dtype=np.uint32)

def get_candidate_regions_to_merge(
    primary_regions: list, 
    candidate_regions: list, 
    mode: MergeMode | int, 
    iou_threshold: float = 0.7, 
    non_overlap_threshold: float = 0.6
) -> list:
    """
    Return candidate regions that should be merged into primary regions.
    
    Args:
        primary_regions: list of regions from the first proposal
        candidate_regions: list of regions to potentially merge
        mode: merge strategy
        iou_threshold: IoU threshold to consider two regions covering the same region
        non_overlap_threshold: threshold for considering a region2 not overlapping with region1
    Returns:
        candidate_regions_to_add: list of regions from candidate_regions that should be merged into primary_regions
    """
    if len(primary_regions) == 0:
        return candidate_regions
    if len(candidate_regions) == 0:
        return []

    primary_masks = get_masks(primary_regions)
    candidate_masks = get_masks(candidate_regions)
    primary_areas = np.sum(primary_masks, axis=(1, 2))
    candidate_areas = np.sum(candidate_masks, axis=(1, 2))

    intersection = get_intersection_mask(primary_masks, candidate_masks)
    union = primary_areas[:, None] + candidate_areas[None, :] - intersection
    iou = intersection / union
    max_iou_for_candidates = np.max(iou, axis=0)

    # indices of candidate regions that share the same region in primary regions
    same_indices = np.where(max_iou_for_candidates > iou_threshold)[0]

    # indices of candidate regions that don't overlap with primary regions
    max_overlap_with_candidate = np.max(intersection / candidate_areas, axis=0)
    non_overlap_indices = np.where(max_overlap_with_candidate < non_overlap_threshold)[0].tolist()

    area_threshold = np.percentile(primary_areas, 50)
    area_candidates_threshold = np.percentile(candidate_areas, 75)
    big_in_primary_indices = np.where(candidate_areas > area_threshold)[0].tolist()
    big_in_candidates_indices = np.where(candidate_areas > area_candidates_threshold)[0].tolist()

    if mode == MergeMode.NON_OVERLAP_OR_AREA:
        final_indices = list(set(non_overlap_indices + big_in_primary_indices + big_in_candidates_indices))
    elif mode == MergeMode.NON_OVERLAP_ONLY:
        final_indices = non_overlap_indices
    elif mode == MergeMode.AREA_ONLY:
        final_indices = big_in_primary_indices
    elif mode == MergeMode.NON_OVERLAP_AND_AREA:
        final_indices = list(set(non_overlap_indices) & set(big_in_primary_indices))
    elif mode == MergeMode.OVERLAP_ONLY:
        final_indices = same_indices
    elif mode == MergeMode.NOT_SAME:
        final_indices = [i for i in range(len(candidate_regions))]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    if mode != MergeMode.OVERLAP_ONLY:
        final_indices = [i for i in final_indices if i not in same_indices]

    final_indices = sorted(final_indices)
    return [candidate_regions[i] for i in final_indices]

def remove_union_regions(regions: list) -> list:
    """
    Remove regions that are effectively unions of other regions.
    
    Args:
        regions: List of region proposals with segmentation masks
        
    Returns:
        List of filtered regions
    """
    if len(regions) == 0:
        return regions
    
    # Convert regions to detections format and get masks
    overlap_ratio_threshold = 0.9
    detections = detections_from_sam(regions)
    masks = detections.mask  # Shape: [N x H x W]
    areas = np.sum(masks, axis=(1, 2))  # Shape: [N]

    def find_containing_regions() -> list[list[int]]:
        """Find regions that contain another."""

        # Calculate pairwise intersection areas between all masks
        intersection = get_intersection_mask(masks, masks) # Shape: [N x N]

        # Amount of overlap between regions normalized 0-1
        overlap_ratios = intersection / areas # Shape: [N x N]

        # For each region, find other indices where overlap ratio > 0.9, except self-overlap
        overlapping_indices = [
            list(set(np.where(overlap_ratios[i] > overlap_ratio_threshold)[0]) - {i})
            for i in range(len(overlap_ratios))
        ]
        return overlapping_indices

    def is_union_of_others(region_idx: int, overlap_indices: list[int]) -> bool:
        """
        Check if current region is effectively a union of other overlapping regions in overlap_indices by:
            - 2 or more regions overlap with the current region
            - IoU between current region and union of other regions > 0.9
        """
        if len(overlap_indices) < 2:  # Need at least 2 regions to form a union
            return False
            
        current_mask = masks[region_idx] # Shape: [H x W]
        current_mask_area = areas[region_idx]
        overlap_masks = masks[overlap_indices] # Shape: [K x H x W]
        union_mask = np.logical_or.reduce(overlap_masks) # Union of other masks. Shape: [H x W]
        
        # Calculate IoU between current region and union of other regions
        intersection = get_intersection_mask(current_mask[np.newaxis, ...], union_mask[np.newaxis, ...]) # Shape: [1 x 1]
        intersection = intersection[0][0]
        iou = intersection / current_mask_area
        
        return iou > overlap_ratio_threshold

    # Find containing regions and filter out unions
    containing_regions: list[list[int]] = find_containing_regions() # Shape: [N x K]
    kept_indices = []
    for i, overlap_indices in enumerate(containing_regions):
        if not is_union_of_others(i, overlap_indices):
            kept_indices.append(i)
    
    return [regions[i] for i in kept_indices]

def remove_duplicate_regions(regions: list, mode='mask', iou_threshold=0.7) -> list:
    """ Remove duplicate regions based on IoU threshold. """
    if len(regions) <= 1:
        return regions

    if mode == 'mask':
        masks = get_masks(regions)
        areas = np.sum(masks, axis=(1, 2))
        
        # Compute IoUs using matrix operations
        intersection = get_intersection_mask(masks, masks)  # Shape: [N x N]
        union = areas[:, None] + areas[None, :] - intersection
        
    elif mode == 'box':
        boxes = get_boxes(regions) # Shape: [N x 4]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        intersection = get_intersection_box(boxes, boxes)
        union = areas[:, None] + areas[None, :] - intersection
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # Zero out diagonal elements
    ious: np.ndarray = intersection / union
    np.fill_diagonal(ious, 0)

    # Return unique regions
    keep_indices = np.ones(len(regions), dtype=bool)
    for index, iou in enumerate(ious):
        if not keep_indices[index]:
            continue

        # drop detections with iou > iou_threshold
        condition = (iou > iou_threshold)
        keep_indices = keep_indices & ~condition
    
    return [regions[i] for i in range(len(regions)) if keep_indices[i]]

class SamGroundingDinoRegionGenerator:

    def __init__(
        self, 
        sam_model, 
        grounding_model: GroundingDinoSAM = None, 
        captioner: ObjectCaptioner = None
    ):
        self.sam_mask_generator = SamMaskGenerator(sam_model)
        self.grounding_dino_sam = grounding_model
        self.captioner = captioner

        if grounding_model is not None:
            assert captioner is not None, "Captioner must be provided if grounding_dino_sam is provided"
    
    def generate_regions(
        self, 
        image: Image.Image, 
        region_mode: str = 'merged',
        return_dict: bool = False,
        duplicate_iou_threshold: float = 0.7, # threshold for removing duplicates within regions

        # merge params
        sam_whole_part_merge_iou_threshold: float = 0.7,
        sam_whole_part_merge_non_overlap_threshold: float = 0.6,
        dino_sam_part_merge_iou_threshold: float = 0.8,
        dino_sam_merge_iou_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Generate regions with specified region mode.

        Args:
            image (Image.Image): input image
            region_mode (str): type of regions to return. Defaults to 'merged'.
                IF region_mode is 'merged' or 'grounding_dino',  grounding_dino_sam and captioner must have been initialized.
                - sam_default: return sam default regions (highest confidence among whole/part/subpart)
                - sam_part: return sam part regions
                - sam_whole: return sam whole regions
                - grounding_dino: return grounding dino regions
                - merged: merge sam whole and part regions. 
                  If grounding_dino_sam is provided, merge grounding_dino regions present in sam_part as well to sam merged regions.
            return_dict (bool, optional): return dict with additional metadata. Defaults to False.

            # For merged region_mode, following params can be used:
            duplicate_iou_threshold (float, optional): threshold for removin duplicate grounding dino regions. Defaults to 0.7.
            sam_whole_part_merge_iou_threshold (float, optional): iou threshold to consider if two sam regions are same. Defaults to 0.7.
            sam_whole_part_merge_non_overlap_threshold (float, optional): overlap threshold to consider if one region convers another. Defaults to 0.6.
            dino_sam_part_merge_iou_threshold (float, optional): iou threshold if grounding_dino is the same as sam part regions. Defaults to 0.8.
            dino_sam_merge_iou_threshold (float, optional): iou threshold if grounding_dino same as sam merged regions. Defaults to 0.7.

        Raises:
            ValueError: region_mode is invalid

        Returns:
            list[dict]: list of region proposals
        """

        try:
            region_mode = RegionMode(region_mode)
        except ValueError:
            raise ValueError(f"Invalid region mode: {region_mode}. Available options: {[m.value for m in RegionMode]}")
        output = {}
        if region_mode == RegionMode.SAM_DEFAULT:
            regions = self.sam_mask_generator.generate_regions(image, 'sam_default')
        elif region_mode == RegionMode.SAM_PART:
            regions = self.sam_mask_generator.generate_regions(image, 'sam_part')
        elif region_mode == RegionMode.SAM_WHOLE:
            regions = self.sam_mask_generator.generate_regions(image, 'sam_whole')
        elif region_mode == RegionMode.GROUNDING_DINO:
            regions, metadata = self.generate_grounding_dino_regions(
                image, 
                remove_duplicates=True, 
                iou_threshold=duplicate_iou_threshold,
            )
            output.update(metadata)
        elif region_mode == RegionMode.MERGED:
            regions_sam_whole = self.sam_mask_generator.generate_regions(image, 'whole')
            regions_sam_whole = remove_union_regions(regions_sam_whole)
            regions_sam_part = self.sam_mask_generator.generate_regions(image, 'part')

            regions_sam_part_to_merge = get_candidate_regions_to_merge(
                regions_sam_whole, regions_sam_part, MergeMode.NON_OVERLAP_OR_AREA,
                iou_threshold=sam_whole_part_merge_iou_threshold,
                non_overlap_threshold=sam_whole_part_merge_non_overlap_threshold
            )
            regions_merged = regions_sam_whole + regions_sam_part_to_merge

            if self.grounding_dino_sam is not None:

                # Generate grounding_dino regions to potentially merge
                regions_dino, metadata = self.generate_grounding_dino_regions(
                    image, 
                    remove_duplicates=True, 
                    iou_threshold=duplicate_iou_threshold
                )
                output.update(metadata)

                # Get grounding_dino regions that overlap with sam part regions
                regions_dino_in_sam_part = get_candidate_regions_to_merge(
                    regions_sam_part, regions_dino, MergeMode.OVERLAP_ONLY, 
                    iou_threshold=dino_sam_part_merge_iou_threshold
                )
                # Then, add the grounding_dino regions excpet similar IoU to any one of regions_merged
                regions_dino_to_merge = get_candidate_regions_to_merge(
                    regions_merged, regions_dino_in_sam_part, MergeMode.NOT_SAME, 
                    iou_threshold=dino_sam_merge_iou_threshold
                )

                # TODO: more merge strategies can be added here
                regions_merged = regions_merged + regions_dino_to_merge
                regions_merged = remove_union_regions(regions_merged)
            
            regions = regions_merged
        else:
            raise ValueError(f"Invalid region mode: {region_mode}. Available region modes: {RegionMode.__members__}")
        
        if not return_dict:
            return regions
        
        output['regions'] = regions
        return output

    
    def generate_grounding_dino_regions(self, image: Image.Image, remove_duplicates=True, iou_threshold:float=0.7) -> list[dict]:
        assert self.grounding_dino_sam is not None, "grounding model not provided"
        assert self.captioner is not None, "Captioner not provided"

        output = self.captioner.generate(image)
        objects = output['objects']
        regions_dino = self.grounding_dino_sam.generate_regions(image, objects)
        
        if remove_duplicates:
            regions_dino = remove_duplicate_regions(regions_dino, mode='mask', iou_threshold=iou_threshold)

        return regions_dino, output