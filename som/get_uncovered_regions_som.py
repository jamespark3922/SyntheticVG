import pickle
import os
from tqdm import tqdm
from task_adapter.seem.tasks import som_w_scaled_bb, som_w_seg
from PIL import Image
import json


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


def main(iou_threshold=0.5, relation_iou_threshold=0.3, sam_folder="data/gqa/som/semantic-sam_slider_1.5", debug=False):
    ann_file = "data/gqa/val_balanced_gqa_GQA_test_200_coco_captions_region_captions_scene_graphs_aokvqa.jsonl"
    # ann_file = "data/gqa/train_balanced_gqa_coco_captions_region_captions_scene_graphs_aokvqa.jsonl"
    annotations = [json.loads(line.strip()) for line in open(ann_file, "r")]
    if debug:
        annotations = annotations[:100]
    image_ids = [annotation["vg_id"] for annotation in annotations]
    output_folder = f"{sam_folder}_uncovered_regions_{iou_threshold}"
    gt_folder = "data/gqa/som/gt_Box_Mark"
    image_folder = "data/vg"
    os.makedirs(output_folder, exist_ok=True)
    average_uncovered = 0
    average_missing_relations = 0
    for image_id in tqdm(image_ids):
        semantic_sam = pickle.load(open(f"{sam_folder}/{image_id}.pkl", "rb"))
        gt = pickle.load(open(f"{gt_folder}/{image_id}.pkl", "rb"))
        print("total SAM proposal", len(semantic_sam))
        print("total GT proposal", len(gt))
        uncovered_regions = []
        uncovered_segs = []
        for item_idx, item in tqdm(enumerate(semantic_sam)):
            bb = item["bbox"]
            x, y, w, h = bb
            bb = [x, y, x+w, y+h]
            found_match = False
            for gt_item_idx, gt_item in tqdm(enumerate(gt), desc=f"item {item_idx}", total=len(gt)):
                gt_bb = gt_item["bbox"]
                iou_score = iou(bb, gt_bb)
                if iou_score > iou_threshold:
                    found_match = True
                    break
            if not found_match:
                new_item = {k:v for k, v in item.items()}
                new_item["bbox"] = bb
                uncovered_regions.append(new_item)
                uncovered_segs.append(item)
        print("uncovered", len(uncovered_regions))
        if len(uncovered_regions) == 0:
            print("no uncovered region")
            continue
        average_uncovered += len(uncovered_regions)
        pickle.dump(uncovered_regions, open(f"{output_folder}/{image_id}.pkl", "wb"))
        visualize_gt = visualize(f"{image_folder}/{image_id}.jpg", gt)
        output_image_file = f"{output_folder}/{image_id}_gt.jpg"
        if visualize_gt is not None:
            # output is a numpy array
            # save output as a jpg file
            output = Image.fromarray(visualize_gt)
            output.save(output_image_file)
        
        visualize_uncovered = visualize(f"{image_folder}/{image_id}.jpg", uncovered_regions)
        output_image_file = f"{output_folder}/{image_id}_uncovered.jpg"
        if visualize_uncovered is not None:
            # output is a numpy array
            # save output as a jpg file
            output = Image.fromarray(visualize_uncovered)
            output.save(output_image_file)
        
        visualize_uncovered = visualize_som(f"{image_folder}/{image_id}.jpg", uncovered_segs)
        output_image_file = f"{output_folder}/{image_id}_uncovered_seg.jpg"
        if visualize_uncovered is not None:
            # output is a numpy array
            # save output as a jpg file
            output = Image.fromarray(visualize_uncovered)
            output.save(output_image_file)
        if len(uncovered_regions) > 0:
            import shutil
            shutil.copyfile(f"{sam_folder}/{image_id}.jpg", f"{output_folder}/{image_id}_sam.jpg")
        missing_relations = 0
        all_regions = list(uncovered_regions+gt)
        sampled_relation_idx = set()
        for item_idx, item in enumerate(uncovered_regions):
            for a_idx, anchor in enumerate(all_regions):
                iou_score = iou(item["bbox"], anchor["bbox"])
                if iou_score >= relation_iou_threshold and iou_score < iou_threshold:
                    missing_relations += 1
                    sampled_relation_idx.add((item_idx, a_idx))
        average_missing_relations += missing_relations
        if len(sampled_relation_idx) == 0:
            continue
        import random
        item_idx, a_idx = random.choice(list(sampled_relation_idx))
        relation_bbs = [uncovered_regions[item_idx], all_regions[a_idx]]
        visualize_relations = visualize(f"{image_folder}/{image_id}.jpg", relation_bbs)
        output_image_file = f"{output_folder}/{image_id}_{item_idx}_{a_idx}_relations.jpg"
        if visualize_relations is not None:
            # output is a numpy array
            # save output as a jpg file
            output = Image.fromarray(visualize_relations)
            output.save(output_image_file)
            print(output_image_file)
    print("average uncovered", average_uncovered/len(image_ids))
    print("average missing relations", average_missing_relations/len(image_ids))


def visualize(image_file, scaled_bbs, label_mode ="1", anno_mode=['Box', 'Mark'], ):
    image = Image.open(image_file)

    # if image has only two dimensions (H, W), add one dimension (C)
    if len(image.size) == 2:
        image = image.convert('RGB')

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'

    output = som_w_scaled_bb(image, scaled_bbs, text_size, label_mode, anno_mode)
    return output


def visualize_som(image_file, seg, label_mode ="1", alpha=0.1, anno_mode=['Mask', 'Mark'], ):
    image = Image.open(image_file)

    # if image has only two dimensions (H, W), add one dimension (C)
    if len(image.size) == 2:
        image = image.convert('RGB')

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'

    output = som_w_seg(image, seg, text_size, label_mode, alpha, anno_mode)
    return output


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
