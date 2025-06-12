import numpy as np
import supervision as sv
from copy import deepcopy
import cv2
from pycocotools import mask as maskUtils
from PIL import Image, ImageDraw
from supervision.detection.utils import mask_to_polygons


def detections_from_sam(regions: list[dict], include_mask=True) -> sv.Detections:
    """
    Convert Segment Anything Model (SAM) region format to supervision Detections format.
    
    Args:
        regions: List of region dictionaries, each containing 'bbox' and optionally 'segmentation'.
                'bbox' should be in [x, y, width, height] format.
        include_mask: Whether to include segmentation masks. Default is True.
        
    Returns:
        sv.Detections: Supervision Detections object containing the converted regions.
                       Returns empty Detections if no regions are provided.
    """

    if regions is None or len(regions) == 0:
        return sv.Detections.empty()
    
    # Convert bounding boxes from [x, y, width, height] to [x1, y1, x2, y2] format
    if len(regions) > 0 and 'xyxy' in regions[0]:
        boxes = np.array([region['xyxy'] for region in regions])
    else:
        boxes = np.array([xywh2xyxy(region['bbox']) for region in regions])

    segmentations = None

    # Process segmentation masks if requested
    if include_mask:
        segmentations = np.array([annToMask(region['segmentation']) for region in regions])
        
    return sv.Detections(xyxy=boxes, mask=segmentations)

def xywh2xyxy(xywh):
    """
    Convert bounding box from [x, y, width, height] to [x1, y1, x2, y2] format.
    
    Args:
        xywh: List or array containing [x, y, width, height]
        
    Returns:
        List containing [x1, y1, x2, y2] coordinates
    """
    x,y,w,h = xywh
    return [x, y, x+w, y+h]

def annToMask(mask_ann, h=None, w=None):
    """
    Convert annotation mask to binary mask.
    
    This function handles different mask formats:
    - Already decoded numpy array masks
    - List of polygons (COCO polygon format)
    - RLE (Run Length Encoding) formats (both compressed and uncompressed)
    
    Args:
        mask_ann: Mask annotation in various formats (numpy array, polygon list, or RLE)
        h: Height of the output mask (optional, used with RLE)
        w: Width of the output mask (optional, used with RLE)
        
    Returns:
        numpy.ndarray: Binary mask as boolean numpy array
    """
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

class SVVisualizer:
    """
    A comprehensive image visualizer for object detection and segmentation results.
    
    This class provides methods to visualize bounding boxes, segmentation masks, 
    polygons, and labels on images. It uses the Supervision library for annotations
    and provides flexibility in controlling the visual aspects of each element.
    
    Attributes:
        bbox_annotator: Annotator for drawing bounding boxes
        mask_annotator: Annotator for drawing segmentation masks
        polygon_annotator: Annotator for drawing polygon outlines
        label_annotator: Annotator for drawing text labels
        top_padding: Additional padding at the top of the image
        right_padding: Additional padding at the right of the image
    """
    def __init__(self, 
                 bbox_thickness: int = 2, 
                 mask_opacity: float = 0.5,
                 label_text_scale: float = 0.5,
                 label_text_thickness: int = 1,
                 label_text_padding: int = 10,
                 label_text_position: str = "top_center", # ["top_left", "center_of_mass", "top_center"]
                 top_padding: int = 0,
                 right_padding: int = 0,
    ):
        """
        Initialize the SVVisualizer with customizable visualization parameters.
        
        Args:
            bbox_thickness: Thickness of bounding box lines. Default is 2.
            mask_opacity: Opacity of segmentation masks (0.0 to 1.0). Default is 0.5.
            label_text_scale: Scale factor for label text size. Default is 0.5.
            label_text_thickness: Thickness of label text. Default is 1.
            label_text_padding: Padding around label text in pixels. Default is 10.
            label_text_position: Position of label text. Options are: "top_left", 
                                "center_of_mass", "top_center". Default is "top_center".
            top_padding: Additional white padding at the top of the image. Default is 0.
            right_padding: Additional white padding at the right of the image. Default is 0.
        """
        self.bbox_annotator = sv.BoxAnnotator(thickness=bbox_thickness, 
                                                      color_lookup=sv.ColorLookup.CLASS)
        self.mask_annotator = sv.MaskAnnotator(opacity=mask_opacity,
                                               color_lookup=sv.ColorLookup.CLASS
                                               )
        self.polygon_annotator = sv.PolygonAnnotator(thickness=bbox_thickness, color_lookup=sv.ColorLookup.CLASS)
        # convert label_text_position to sv.Position
        label_text_position = getattr(sv.Position, label_text_position.upper())
        self.label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.CLASS,
            text_scale=label_text_scale,
            text_thickness=label_text_thickness,
            text_padding=label_text_padding,
            text_position=label_text_position,
        )
        
        self.top_padding = top_padding
        self.right_padding = right_padding
        
    def draw_masks(self, 
                   image_rgb: np.ndarray, 
                   detections: sv.Detections, 
                   labels: list[str] = None,
                   draw_bbox: bool = True, 
                   draw_polygon: bool = False,
                   draw_mask: bool = True, 
                   draw_label: bool = True,
                   reverse_order=False,
    ) -> np.ndarray:
        """
        Draw detection elements (boxes, masks, polygons, labels) on an image.
        
        Args:
            image_rgb: Input RGB image as numpy array (HxWx3)
            detections: Supervision Detections object containing detection data
            labels: List of strings for object labels. If None, uses numeric indices.
            draw_bbox: Whether to draw bounding boxes. Default is True.
            draw_polygon: Whether to draw polygon outlines. Default is False.
            draw_mask: Whether to draw segmentation masks. Default is True.
            draw_label: Whether to draw text labels. Default is True.
            reverse_order: Whether to reverse the order of drawing detections.
                          Useful for handling occlusions differently. Default is False.
        
        Returns:
            np.ndarray: The annotated image as RGB numpy array
        """
        image_display = image_rgb.copy() # cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        detections = deepcopy(detections)
        detections.class_id = np.arange(len(detections))
        if labels is None:
            labels = [str(i) for i in range(len(detections))]
        if reverse_order:
            detections = detections[::-1]
            labels = labels[::-1]
            
        if draw_bbox:
            image_display = self.bbox_annotator.annotate(image_display, detections)
        if draw_mask:
            image_display = self.mask_annotator.annotate(image_display, detections)
        if draw_polygon:
            image_display = self.polygon_annotator.annotate(image_display, detections)
        if draw_label: # draw label on top of image, optionally with padding
            has_padding = self.top_padding or self.right_padding
            if self.top_padding: 
                # need to shift detections down
                detections = deepcopy(detections)
                detections.xyxy[:,1] += self.top_padding
            if has_padding:
                h,w = image_rgb.shape[:2]
                # paste image_rgb onto blank_image to [0, top_padding] position
                blank_image = Image.new('RGB', (w+self.right_padding, h+self.top_padding), (255, 255, 255))
                blank_image.paste(Image.fromarray(image_display), (0, self.top_padding))
                image_display = np.array(blank_image)
                
            image_display = self.label_annotator.annotate(image_display, detections, labels)
        
        return image_display
    
    def insert_white_padding(self, image_rgb: np.ndarray) -> np.ndarray:
        h,w = image_rgb.shape[:2]
        image_with_padding = np.ones((h+self.top_padding, w+self.right_padding, 3), dtype=np.uint8) * 255
        image_with_padding[:h, :w] = image_rgb
        return image_with_padding 

def create_visualizer_for_image(image: np.ndarray, 
                              label_text_position: str = "top_left",
                              white_padding: int = 0,
                              bbox_thickness: int = None,
                              mask_opacity: float = 0.5,
                              custom_scaling: dict = None) -> SVVisualizer:
    """
    Create and configure a visualizer with parameters automatically scaled based on image size.
    
    This function creates an SVVisualizer instance with visualization parameters (text size,
    line thickness, etc.) that are appropriate for the given image dimensions.
    
    Args:
        image: Input image as numpy array (HxWx3)
        label_text_position: Position of label text. Options are: "top_left", 
                            "center_of_mass", "top_center". Default is "top_left".
        white_padding: Amount of white padding to add around the image. Default is 0.
        bbox_thickness: Thickness of bounding box lines. If None, will be auto-scaled.
        mask_opacity: Opacity of segmentation masks (0.0 to 1.0). Default is 0.5.
        custom_scaling: Optional dictionary to override default scaling factors.
            - 'padding_divisor': Divisor for label padding (default: 100)
            - 'thickness_divisor': Divisor for text thickness (default: 500)
            - 'scale_divisor': Divisor for text scale (default: 1000)
            - 'min_scale': Minimum text scale (default: 0.5)
                       
    Returns:
        SVVisualizer: Configured visualizer object
    """
    # Get image dimensions
    h, w = image.shape[:2]
    largest_edge = max(h, w)
    
    # Default scaling factors
    scaling = {
        'padding_divisor': 100,
        'thickness_divisor': 1000,
        'scale_divisor': 1000,
        'min_scale': 0.5
    }
    
    # Override with custom scaling if provided
    if custom_scaling:
        scaling.update(custom_scaling)
    
    # Auto-scale parameters based on image size
    label_text_padding = largest_edge // scaling['padding_divisor']
    label_text_thickness = max(1, largest_edge // scaling['thickness_divisor'])
    label_text_scale = max(scaling['min_scale'], largest_edge // scaling['scale_divisor'])
    
    # Use provided bbox_thickness or auto-scale it
    if bbox_thickness is None:
        bbox_thickness = max(1, int(largest_edge / 1000) * 2)  # Scale with image size, minimum 1
    
    # Create and return the visualizer
    return SVVisualizer(
        bbox_thickness=bbox_thickness,
        mask_opacity=mask_opacity,
        label_text_padding=label_text_padding,
        label_text_thickness=label_text_thickness,
        label_text_scale=label_text_scale,
        label_text_position=label_text_position,
        top_padding=white_padding,
        right_padding=white_padding,
    )


def visualize_masks(image_rgb: np.ndarray, 
                   regions: list[dict] | sv.Detections, 
                   labels: list[str] = None,
                   draw_bbox: bool = True, 
                   draw_mask: bool = True, 
                   draw_polygon: bool = False, 
                   draw_label: bool = True,
                   label_text_position: str = "top_left",
                   white_padding: int = 0,
                   reverse_order: bool = False,
                   plot_image: bool = False,
                   visualizer: SVVisualizer = None,
                   custom_scaling: dict = None) -> np.ndarray:
    """
    High-level function to visualize object detections with masks, boxes, and labels.
    
    This function provides a convenient interface to visualize detection results
    by automatically configuring the visualizer based on image size and user preferences.
    
    Args:
        image_rgb: Input RGB image as numpy array (HxWx3)
        regions: Either a list of region dictionaries (SAM format) or a supervision Detections object
        labels: List of strings for object labels. If None, uses numeric indices.
        draw_bbox: Whether to draw bounding boxes. Default is True.
        draw_mask: Whether to draw segmentation masks. Default is True.
        draw_polygon: Whether to draw polygon outlines. Default is False.
        draw_label: Whether to draw text labels. Default is True.
        label_text_position: Position of label text. Options are: "top_left", 
                            "center_of_mass", "top_center". Default is "top_left".
        white_padding: Amount of white padding to add around the image. Default is 0.
        reverse_order: Whether to reverse the order of drawing detections.
                      Useful for handling occlusions differently. Default is False.
        plot_image: Whether to display the image using supervision's plot_image.
                   Default is False.
        visualizer: Optional pre-configured SVVisualizer instance. If None,
                   a new one will be created with parameters based on image size.
        custom_scaling: Optional dictionary to override default scaling factors
                       when creating the visualizer.
                   
    Returns:
        np.ndarray: The annotated image as RGB numpy array
    """
    # Convert regions to supervision Detections if needed
    if isinstance(regions, sv.Detections):
        detections = regions
    else:
        detections = detections_from_sam(regions, include_mask=draw_mask or draw_polygon)
    
    # Convert color space if needed
    image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Create a visualizer if not provided
    if visualizer is None:
        visualizer = create_visualizer_for_image(
            image_rgb,
            label_text_position=label_text_position,
            white_padding=white_padding,
            custom_scaling=custom_scaling
        )
    
    # Draw all requested elements
    image_display = visualizer.draw_masks(
        image_display, 
        detections,
        labels=labels,
        draw_bbox=draw_bbox, 
        draw_mask=draw_mask, 
        draw_polygon=draw_polygon,
        draw_label=draw_label,
        reverse_order=reverse_order
    )
    
    # Convert back to RGB color space
    image_with_masks = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)

    # Optionally display the image
    if plot_image:
        sv.plot_image(image_display)
    
    return image_with_masks

def visualize_masks(image_rgb: np.ndarray, regions: list[dict] | sv.Detections, 
                    labels: list[str] = None,
                    draw_bbox: bool = True, draw_mask: bool = True, draw_polygon: bool = False, draw_label: bool = True,
                    label_text_position: str = "top_left",
                    white_padding: int = 0,
                    reverse_order: bool = False,
                    plot_image: bool = False) -> np.ndarray:   
    """
    High-level function to visualize object detections with masks, boxes, and labels.
    
    This function provides a convenient interface to visualize detection results
    by automatically configuring the visualizer based on image size and user preferences.
    
    Args:
        image_rgb: Input RGB image as numpy array (HxWx3)
        regions: Either a list of region dictionaries (SAM format) or a supervision Detections object
        labels: List of strings for object labels. If None, uses numeric indices.
        draw_bbox: Whether to draw bounding boxes. Default is True.
        draw_mask: Whether to draw segmentation masks. Default is True.
        draw_polygon: Whether to draw polygon outlines. Default is False.
        draw_label: Whether to draw text labels. Default is True.
        label_text_position: Position of label text. Options are: "top_left", 
                            "center_of_mass", "top_center". Default is "top_left".
        white_padding: Amount of white padding to add around the image. Default is 0.
        reverse_order: Whether to reverse the order of drawing detections.
                      Useful for handling occlusions differently. Default is False.
        plot_image: Whether to display the image using supervision's plot_image.
                   Default is False.
                   
    Returns:
        np.ndarray: The annotated image as RGB numpy array
    """
    if isinstance(regions, sv.Detections):
        detections = regions
    else:
        detections: sv.Detections = detections_from_sam(regions, include_mask=draw_mask or draw_polygon)
    image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h,w = image_rgb.shape[:2]
    largest_edge = max(h,w)
    visualizer = SVVisualizer(
        label_text_padding=largest_edge//100,
        label_text_thickness=largest_edge//500,
        label_text_scale=max(0.5, largest_edge//1000),
        label_text_position=label_text_position,
        top_padding=white_padding,
        right_padding=white_padding,
    )
    image_display: np.ndarray = visualizer.draw_masks(image_display, detections,
                                                      labels=labels,
                                                      draw_bbox=draw_bbox, 
                                                      draw_mask=draw_mask, 
                                                      draw_polygon=draw_polygon,
                                                      draw_label=draw_label,
                                                      reverse_order=reverse_order
                                                      )  
    image_with_masks = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)

    if plot_image:
        sv.plot_image(image_display)
    
    return image_with_masks