

## How to Run

#python evaluation/evaluate_images.py \
#  /path/to/OUTDIR \
#  --outfile results/results.jsonl \
#  --model-config /path/to/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
#  --model-path   /path/to/checkpoints \
#  --options threshold=0.3 max_objects=16 clip_model=ViT-L-14 bgcolor=#999

##===============================================================================================











#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate generated images using Mask2Former (or other MMDetection 3.x model).
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

from mmdet.apis import inference_detector, init_detector
import mmdet

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it  # silence tqdm

# ----------------------------
# Globals (filled in main)
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise SystemExit("CUDA device required (torch.cuda.is_available() == False).")

OBJECT_DETECTOR = None
CLASSNAMES = None

CLIP_MODEL = None
CLIP_TRANSFORM = None
CLIP_TOKENIZER = None

THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}


# ----------------------------
# Args
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str, help="Root folder containing <id>/metadata.jsonl and /samples/*.png")
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, required=False,
                        help="Path to MMDetection config .py (REQUIRED in MMDet 3.x; no packaged configs).")
    parser.add_argument("--model-path", type=str, default="./",
                        help="Folder containing checkpoint, or a direct .pth path.")
    parser.add_argument("--options", nargs="*", type=str, default=[],
                        help="key=value pairs, e.g. threshold=0.3 max_objects=16 clip_model=ViT-L-14 bgcolor=#999")
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)

    # MMDet 3.x does not ship configs in the wheel; require a local config
    if args.model_config is None:
        # try to auto-discover a config in --model-path
        if os.path.isdir(args.model_path):
            cands = [os.path.join(args.model_path, f) for f in os.listdir(args.model_path) if f.endswith(".py")]
            if cands:
                args.model_config = cands[0]
            else:
                raise SystemExit(
                    "No --model-config provided and no .py config found in --model-path.\n"
                    "Clone mmdetection and pass the config explicitly via --model-config."
                )
        elif os.path.isfile(args.model_path) and args.model_path.endswith(".py"):
            args.model_config = args.model_path
        else:
            raise SystemExit("Please provide --model-config pointing to a local MMDetection config .py.")
    return args


# ----------------------------
# Utilities
# ----------------------------
def timed(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        print(f"[timing] {fn.__name__} took {time.time() - t0:.3f}s", file=sys.stderr)
        return out
    return wrapper


def _resolve_ckpt_path(model_path_opt: str, default_name: str) -> str:
    """Return a checkpoint .pth path given --model-path and a default model name."""
    if os.path.isfile(model_path_opt) and model_path_opt.endswith(".pth"):
        return model_path_opt
    if os.path.isdir(model_path_opt):
        # prefer <default_name>.pth if present; else pick the first .pth
        candidate = os.path.join(model_path_opt, f"{default_name}.pth")
        if os.path.isfile(candidate):
            return candidate
        pths = [os.path.join(model_path_opt, f) for f in os.listdir(model_path_opt) if f.endswith(".pth")]
        if pths:
            return pths[0]
    raise SystemExit(f"Could not locate checkpoint .pth in --model-path: {model_path_opt}")


# ----------------------------
# Load models
# ----------------------------
@timed
def load_models(args):
    global OBJECT_DETECTOR, CLASSNAMES, CLIP_MODEL, CLIP_TRANSFORM, CLIP_TOKENIZER

    config_path = args.model_config
    model_key = args.options.get('model', "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco")
    ckpt_path = _resolve_ckpt_path(args.model_path, model_key)

    OBJECT_DETECTOR = init_detector(config_path, ckpt_path, device=DEVICE)

    # Class names: prefer dataset_meta from the model; fallback to object_names.txt
    if hasattr(OBJECT_DETECTOR, "dataset_meta") and OBJECT_DETECTOR.dataset_meta and \
       "classes" in OBJECT_DETECTOR.dataset_meta:
        CLASSNAMES = list(OBJECT_DETECTOR.dataset_meta["classes"])
    else:
        cls_file = os.path.join(os.path.dirname(__file__), "object_names.txt")
        if not os.path.isfile(cls_file):
            raise SystemExit("Could not infer class names; put object_names.txt next to this script.")
        with open(cls_file) as f:
            CLASSNAMES = [line.strip() for line in f]

    clip_arch = args.options.get('clip_model', "ViT-L-14")
    CLIP_MODEL, _, CLIP_TRANSFORM = open_clip.create_model_and_transforms(
        clip_arch, pretrained="openai", device=DEVICE
    )
    CLIP_TOKENIZER = open_clip.get_tokenizer(clip_arch)


# ----------------------------
# Color classification helper
# ----------------------------
class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, bgcolor: str = "#999", do_crop: bool = True):
        self._image = image.convert("RGB")
        self._blank = self._image.copy() if bgcolor == "original" else Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._do_crop = do_crop

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)
            img = Image.composite(self._image, self._blank, Image.fromarray(mask * 255))
        else:
            img = self._image
        if self._do_crop:
            img = img.crop(box[:4])
        return (CLIP_TRANSFORM(img), 0)


def color_classification(image, bboxes, classname, device=DEVICE, bgcolor="#999", do_crop=True):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            CLIP_MODEL, CLIP_TOKENIZER, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            device
        )
    clf = COLOR_CLASSIFIERS[classname]
    ds = ImageCrops(image, bboxes, bgcolor=bgcolor, do_crop=do_crop)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)
    with torch.no_grad():
        pred, _ = zsc.run_classification(CLIP_MODEL, clf, dl, device)
        return [COLORS[i.item()] for i in pred.argmax(1)]


# ----------------------------
# Geometry & relations
# ----------------------------
def compute_iou(box_a, box_b):
    area = lambda b: max(b[2] - b[0] + 1, 0) * max(b[3] - b[1] + 1, 0)
    i_area = area([max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
                   min(box_a[2], box_b[2]), min(box_a[3], box_b[3])])
    u_area = area(box_a) + area(box_b) - i_area
    return i_area / u_area if u_area else 0.0


def relative_position(obj_a, obj_b):
    """Position of A relative to B; returns a set of relations."""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / (np.linalg.norm(offset) + 1e-9)
    rels = set()
    if dx < -0.5: rels.add("left of")
    if dx > 0.5:  rels.add("right of")
    if dy < -0.5: rels.add("above")
    if dy > 0.5:  rels.add("below")
    return rels


def evaluate(image, objects, metadata, bgcolor="#999", do_crop=True):
    """
    include: AND across clauses; exclude: OR across clauses.
    Color/position evaluated on top-K confident objects per class.
    """
    correct = True
    reason = []
    matched_groups = []

    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found = objects.get(classname, [])[:req['count']]
        if len(found) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found)}")
        else:
            if 'color' in req:
                colors = color_classification(image, found, classname, device=DEVICE,
                                              bgcolor=bgcolor, do_crop=do_crop)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    counts = ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    reason.append(f"expected {req['color']} {classname}>={req['count']}, "
                                  f"found {colors.count(req['color'])} {req['color']}; and {counts}")
            if 'position' in req and matched:
                expected_rel, target_group = req['position']
                if target_group >= len(matched_groups) or matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found:
                        ok = False
                        for tgt in matched_groups[target_group]:
                            if expected_rel in relative_position(obj, tgt):
                                ok = True; break
                        if not ok:
                            correct = matched = False
                            reason.append(f"expected {classname} {expected_rel} target")
                            break
        matched_groups.append(found if matched else None)

    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")

    return correct, "\n".join(reason)


# ----------------------------
# MMDet 3.x result adapter
# ----------------------------
def _to_numpy_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().to("cpu").numpy()
    else:
        arr = np.asarray(mask)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8)
    return arr


def parse_mmdet_output(result):
    """
    Convert DetDataSample → dict[class] -> list[(bbox[0:5], mask_or_None)].
    bbox: [x1,y1,x2,y2,score]
    """
    detected = {c: [] for c in CLASSNAMES}

    ds = result
    if not hasattr(ds, "pred_instances"):
        # sometimes wrapped in list
        if isinstance(result, (list, tuple)) and len(result) > 0 and hasattr(result[0], "pred_instances"):
            ds = result[0]
        else:
            return {k: [] for k in CLASSNAMES}

    inst = ds.pred_instances
    bboxes = inst.get("bboxes", None)
    scores = inst.get("scores", None)
    labels = inst.get("labels", None)
    masks  = inst.get("masks", None)

    if bboxes is None or scores is None or labels is None:
        return {k: [] for k in CLASSNAMES}

    bboxes = bboxes.detach().to("cpu").numpy()
    scores = scores.detach().to("cpu").numpy()
    labels = labels.detach().to("cpu").numpy().astype(int)

    if masks is not None:
        if hasattr(masks, "to_tensor"):  # BitmapMasks
            masks_np = masks.to_tensor(dtype=torch.bool, device="cpu").numpy()
        elif isinstance(masks, torch.Tensor):
            masks_np = masks.detach().to("cpu").numpy()
        else:
            masks_np = None
    else:
        masks_np = None

    for i in range(bboxes.shape[0]):
        li = labels[i]
        if li < 0 or li >= len(CLASSNAMES):
            continue
        name = CLASSNAMES[li]
        x1, y1, x2, y2 = bboxes[i].tolist()
        sc = float(scores[i])
        box5 = [x1, y1, x2, y2, sc]
        mask = _to_numpy_mask(masks_np[i]) if masks_np is not None else None
        detected[name].append((box5, mask))

    return {k: v for k, v in detected.items() if v}


# ----------------------------
# Image evaluation
# ----------------------------
def evaluate_image(filepath, metadata):
    result = inference_detector(OBJECT_DETECTOR, filepath)
    image = ImageOps.exif_transpose(Image.open(filepath))

    detected = parse_mmdet_output(result)

    # Thresholding / NMS / Top-K per class
    conf_thr = THRESHOLD if metadata.get('tag') != "counting" else COUNTING_THRESHOLD
    filtered = {}
    for cname, items in detected.items():
        items = sorted(items, key=lambda t: t[0][4], reverse=True)
        items = [it for it in items if it[0][4] > conf_thr]
        kept = []
        for box, mask in items:
            if NMS_THRESHOLD < 1.0 and any(compute_iou(box, kbox) >= NMS_THRESHOLD for kbox, _ in kept):
                continue
            kept.append((box, mask))
            if len(kept) >= MAX_OBJECTS:
                break
        if kept:
            filtered[cname] = kept

    bgcolor = metadata.get('bgcolor', "#999")
    do_crop = (metadata.get('crop', '1') == '1')
    ok, reason = evaluate(image, filtered, metadata, bgcolor=bgcolor, do_crop=do_crop)

    return {
        'filename': filepath,
        'tag': metadata.get('tag'),
        'prompt': metadata.get('prompt'),
        'correct': ok,
        'reason': reason,
        'metadata': json.dumps(metadata),
        'details': json.dumps({k: [b[0] for b in v] for k, v in filtered.items()})
    }


# ----------------------------
# Main
# ----------------------------
def main(args):
    out = []
    for sub in os.listdir(args.imagedir):
        folder = os.path.join(args.imagedir, sub)
        if not os.path.isdir(folder) or not sub.isdigit():
            continue

        meta_path = os.path.join(folder, "metadata.jsonl")
        if not os.path.isfile(meta_path):
            print(f"[warn] missing {meta_path}", file=sys.stderr); continue
        with open(meta_path) as fp:
            first = fp.readline().strip()
            metadata = json.loads(first)

        samples = os.path.join(folder, "samples")
        if not os.path.isdir(samples):
            print(f"[warn] missing samples dir: {samples}", file=sys.stderr); continue

        for name in os.listdir(samples):
            if not re.match(r"\d+\.png", name):
                continue
            img_path = os.path.join(samples, name)
            if not os.path.isfile(img_path):
                continue
            out.append(evaluate_image(img_path, metadata))

    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(out).to_json(fp, orient="records", lines=True)


if __name__ == "__main__":
    args = parse_args()
    # thresholds/options
    if "threshold" in args.options:          THRESHOLD = float(args.options["threshold"])
    if "counting_threshold" in args.options: COUNTING_THRESHOLD = float(args.options["counting_threshold"])
    if "max_objects" in args.options:        MAX_OBJECTS = int(args.options["max_objects"])
    if "max_overlap" in args.options:        NMS_THRESHOLD = float(args.options["max_overlap"])
    if "position_threshold" in args.options: POSITION_THRESHOLD = float(args.options["position_threshold"])

    load_models(args)
    main(args)

