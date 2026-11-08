#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate SAGA stimuli: configuration B, five fixed semantic removal ratios.

Run from the existing SAM2/LaMa project directory (Windows cmd.exe example):
    python semantic_attack_B_5sr.py --input_dir images --output_dir out_B --device cpu

For every readable input image, save five RGB PNGs:
    <stem>_sr0.30B.png
    <stem>_sr0.45B.png
    <stem>_sr0.60B.png
    <stem>_sr0.75B.png
    <stem>_sr0.90B.png

Configuration B is fixed: lambda_a = lambda_c = lambda_f = lambda_s = 0.25.
Regions are removed in ascending priority order, using the closest cumulative
semantic-weight prefix. SAM2 segmentation/filtering/scoring run once per image.
Identical final masks reuse their reconstruction but still save all five files.
If no regions are retained, all five files copy the preprocessed input.

This version exports only images. It does not calculate perceptual metrics or
write Excel files, masks, overlays, or region reports. LPIPS, DreamSim, and
openpyxl are not required. Use the existing SAM2/LaMa Python environment.

The original preprocessing is preserved: RGB conversion and optional resizing
of the longest side to at most 1024 pixels. Use --resize_max_side 0 to retain
input dimensions. Different input sizes are supported; aspect ratio is kept.
Input scanning is non-recursive unless --recursive is supplied. Output subtrees
are excluded, and animated/multipage files use the first frame only.
Use a fresh output directory or --overwrite to replace matching output files.
Errors are printed to the console; remaining inputs/conditions continue.
"""

from __future__ import annotations

import os
import math
import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


# ============================================================
# Data structures
# ============================================================

@dataclass
class Region:
    region_id: int
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    area_pixels: int
    area_ratio: float
    centroid_xy: Tuple[float, float]
    centrality_score: float
    foreground_score: float
    stability_score: float
    priority_score: float
    semantic_weight: float = 0.0
    is_filtered: bool = False
    filter_reason: str = ""


@dataclass
class AttackConfig:
    # filtering
    min_area_ratio: float = 0.005
    max_area_ratio: float = 1.0 / 3.0
    max_regions_after_filter: int = 64
    duplicate_iou_threshold: float = 0.60
    containment_threshold: float = 0.90

    # scoring
    weight_area: float = 0.25
    weight_centrality: float = 0.25
    weight_foreground: float = 0.25
    weight_stability: float = 0.25

    # region selection
    selection_mode: str = "semantic_mass"

    # mask refinement
    dilate_radius: int = 11
    blur_radius: int = 1
    lama_binarize_threshold: int = 16

    # performance / image resize
    resize_max_side: int = 1024

# ============================================================
# I/O helpers
# ============================================================

def load_image(path: str, max_side: int = 1024) -> Image.Image:
    with Image.open(path) as source:
        img = source.convert("RGB")
    if max_side == 0:
        return img
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def save_image(img: Image.Image, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img.save(path)


def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)


def np_mask_to_pil(mask: np.ndarray) -> Image.Image:
    arr = (mask.astype(np.uint8) * 255)
    return Image.fromarray(arr, mode="L")


# ============================================================
# Geometry / mask utilities
# ============================================================

def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def mask_bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def mask_intersection_over_smaller(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = min(a.sum(), b.sum())
    if denom == 0:
        return 0.0
    return float(inter) / float(denom)


def union_masks(masks: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=bool)
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out


def dilate_and_blur_mask(mask: np.ndarray, dilate_radius: int, blur_radius: int) -> Image.Image:
    pil_mask = np_mask_to_pil(mask)

    if dilate_radius > 0:
        size = max(3, 2 * dilate_radius + 1)
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size=size))

    if blur_radius > 0:
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return pil_mask


def format_sr_tag(sr: float) -> str:
    return f"sr{sr:.2f}"


# ============================================================
# Backend A: Automatic segmentation
# ============================================================

class AutoMaskGeneratorBackend:
    def generate(self, image: Image.Image) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DummyGridMaskGenerator(AutoMaskGeneratorBackend):
    def generate(self, image: Image.Image) -> List[Dict[str, Any]]:
        w, h = image.size
        H, W = h, w
        masks = []

        def rect_mask(x1, y1, x2, y2, stab):
            m = np.zeros((H, W), dtype=bool)
            m[y1:y2, x1:x2] = True
            return {"segmentation": m, "stability_score": stab}

        masks.append(rect_mask(int(0.06 * W), int(0.58 * H), int(0.16 * W), int(0.74 * H), 0.88))
        masks.append(rect_mask(int(0.22 * W), int(0.42 * H), int(0.34 * W), int(0.64 * H), 0.85))
        masks.append(rect_mask(int(0.35 * W), int(0.25 * H), int(0.68 * W), int(0.78 * H), 0.92))
        masks.append(rect_mask(int(0.72 * W), int(0.38 * H), int(0.90 * W), int(0.72 * H), 0.81))
        masks.append(rect_mask(int(0.00 * W), int(0.78 * H), int(1.00 * W), int(1.00 * H), 0.60))

        return masks


class SAM2AutoMaskGeneratorBackend(AutoMaskGeneratorBackend):
    def __init__(self, model_cfg: str, checkpoint: str, device: str = "cpu") -> None:
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device
        self._generator = None

    def _lazy_init(self) -> None:
        if self._generator is not None:
            return

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except Exception as e:
            raise RuntimeError(
                "Could not import SAM2. Please verify installation and active environment."
            ) from e

        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available. Falling back to CPU.", flush=True)
            device = "cpu"

        print(f"[INFO] Initializing SAM2 on device={device}", flush=True)
        print(f"[INFO] SAM2 config={self.model_cfg}", flush=True)
        print(f"[INFO] SAM2 checkpoint={self.checkpoint}", flush=True)

        model = build_sam2(self.model_cfg, self.checkpoint, device=device)

        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=12,
            points_per_batch=24,
            pred_iou_thresh=0.72,
            stability_score_thresh=0.90,
            stability_score_offset=1.0,
            crop_n_layers=0,
            box_nms_thresh=0.60,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=400,
            use_m2m=False,
        )

    def generate(self, image: Image.Image) -> List[Dict[str, Any]]:
        self._lazy_init()
        img_np = pil_to_np_rgb(image)
        print("[INFO] SAM2 generating masks...", flush=True)
        masks = self._generator.generate(img_np)
        print(f"[INFO] SAM2 raw masks generated: {len(masks)}", flush=True)

        normalized = []
        for m in masks:
            seg = np.asarray(m["segmentation"]).astype(bool)
            normalized.append({
                "segmentation": seg,
                "stability_score": float(m.get("stability_score", 0.5)),
                "predicted_iou": float(m.get("predicted_iou", 0.5)),
                "area": int(m.get("area", int(seg.sum()))),
                "bbox": m.get("bbox", None),
            })
        return normalized


# ============================================================
# Backend B: Prompt-free inpainting
# ============================================================

class InpaintingBackend:
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        raise NotImplementedError


class DummyInpaintingBackend(InpaintingBackend):
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=20))
        out = Image.composite(blurred, image, mask)
        return out


class SimpleLaMaBackend(InpaintingBackend):
    def __init__(self, binarize_threshold: int = 16, device: str = "cpu") -> None:
        self._model = None
        self.binarize_threshold = binarize_threshold
        self.device = device

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from simple_lama_inpainting import SimpleLama
        except Exception as e:
            raise RuntimeError(
                "Could not import simple_lama_inpainting. Please install it in the active environment."
            ) from e

        print("[INFO] Initializing SimpleLaMa...", flush=True)
        self._model = SimpleLama(device=torch.device(self.device))

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self._lazy_init()

        if mask.mode != "L":
            mask = mask.convert("L")

        mask_np = np.array(mask)
        mask_bin = (mask_np > self.binarize_threshold).astype(np.uint8) * 255
        mask_bin_img = Image.fromarray(mask_bin, mode="L")

        print("[INFO] Running LaMa inpainting...", flush=True)
        result = self._model(image, mask_bin_img)

        if not isinstance(result, Image.Image):
            result = Image.fromarray(np.asarray(result))

        # SimpleLaMa may retain the right/bottom padding to multiples of eight.
        # Remove only that padding; do not resize the reconstructed image.
        if result.size != image.size:
            padded_size = (((image.width + 7) // 8) * 8, ((image.height + 7) // 8) * 8)
            if result.size != padded_size:
                raise ValueError(f"Unexpected LaMa output size: {result.size}; input: {image.size}")
            result = result.crop((0, 0, image.width, image.height))
        return result.convert("RGB")


# ============================================================
# Region scoring
# ============================================================

def compute_centrality_score(centroid_xy: Tuple[float, float], width: int, height: int) -> float:
    cx, cy = centroid_xy
    img_cx, img_cy = width / 2.0, height / 2.0

    dx = (cx - img_cx) / max(width / 2.0, 1e-6)
    dy = (cy - img_cy) / max(height / 2.0, 1e-6)
    d = math.sqrt(dx * dx + dy * dy)

    score = max(0.0, 1.0 - d / math.sqrt(2.0))
    return float(score)


def compute_foreground_score(
    area_ratio: float,
    centrality_score: float,
    bbox_xyxy: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    width, height = image_size
    bbox_w = max(1, x2 - x1 + 1)
    bbox_h = max(1, y2 - y1 + 1)
    bbox_ratio = (bbox_w * bbox_h) / float(width * height)

    size_term = math.exp(-((area_ratio - 0.10) ** 2) / (2 * (0.12 ** 2)))
    bbox_term = math.exp(-((bbox_ratio - 0.18) ** 2) / (2 * (0.18 ** 2)))

    score = 0.55 * centrality_score + 0.25 * size_term + 0.20 * bbox_term
    return float(max(0.0, min(1.0, score)))


def normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def build_regions_from_raw_masks(
    raw_masks: List[Dict[str, Any]],
    image: Image.Image,
    cfg: AttackConfig,
) -> List[Region]:
    width, height = image.size
    H, W = height, width

    temp = []
    for idx, item in enumerate(raw_masks):
        seg = np.asarray(item["segmentation"]).astype(bool)

        area = mask_area(seg)
        area_ratio = area / float(H * W)
        bbox = mask_bbox_xyxy(seg)
        centroid = mask_centroid(seg)
        centrality = compute_centrality_score(centroid, width, height)

        stability = float(item.get("stability_score", item.get("predicted_iou", 0.5)))
        foreground = compute_foreground_score(area_ratio, centrality, bbox, (width, height))

        temp.append({
            "region_id": idx,
            "mask": seg,
            "bbox_xyxy": bbox,
            "area_pixels": area,
            "area_ratio": area_ratio,
            "centroid_xy": centroid,
            "centrality_score": centrality,
            "foreground_score": foreground,
            "stability_score": stability,
        })

    regions: List[Region] = []
    for t in temp:
        regions.append(
            Region(
                region_id=t["region_id"],
                mask=t["mask"],
                bbox_xyxy=t["bbox_xyxy"],
                area_pixels=t["area_pixels"],
                area_ratio=t["area_ratio"],
                centroid_xy=t["centroid_xy"],
                centrality_score=t["centrality_score"],
                foreground_score=t["foreground_score"],
                stability_score=t["stability_score"],
                priority_score=0.0,
            )
        )

    return regions


def score_regions(regions: List[Region], cfg: AttackConfig) -> List[Region]:
    """Compute priority scores and normalized semantic weights after filtering."""
    if not regions:
        return regions

    coefficients = [
        cfg.weight_area,
        cfg.weight_centrality,
        cfg.weight_foreground,
        cfg.weight_stability,
    ]
    if any(weight < 0.0 for weight in coefficients):
        raise ValueError("Priority coefficients must be non-negative.")
    if not math.isclose(sum(coefficients), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Priority coefficients must sum to 1.")

    cent_norm = normalize_scores([r.centrality_score for r in regions])
    fore_norm = normalize_scores([r.foreground_score for r in regions])
    stab_norm = normalize_scores([r.stability_score for r in regions])

    for region, c, f, s in zip(regions, cent_norm, fore_norm, stab_norm):
        # area_ratio already equals |m_i| / (H * W), as defined in the paper.
        region.priority_score = float(
            cfg.weight_area * region.area_ratio
            + cfg.weight_centrality * c
            + cfg.weight_foreground * f
            + cfg.weight_stability * s
        )

    priority_sum = sum(max(0.0, r.priority_score) for r in regions)
    if priority_sum <= 1e-12:
        uniform_weight = 1.0 / len(regions)
        for region in regions:
            region.semantic_weight = uniform_weight
    else:
        for region in regions:
            region.semantic_weight = max(0.0, region.priority_score) / priority_sum

    return regions


# ============================================================
# Filtering / de-duplication
# ============================================================

def filter_regions(regions: List[Region], cfg: AttackConfig) -> List[Region]:
    kept: List[Region] = []

    for r in regions:
        if r.area_ratio < cfg.min_area_ratio:
            r.is_filtered = True
            r.filter_reason = "too_small"
            continue
        if r.area_ratio > cfg.max_area_ratio:
            r.is_filtered = True
            r.filter_reason = "too_large"
            continue
        kept.append(r)

    kept.sort(key=lambda x: (x.stability_score, x.area_ratio), reverse=True)

    deduped: List[Region] = []
    for r in kept:
        duplicate = False
        for q in deduped:
            iou = bbox_iou_xyxy(r.bbox_xyxy, q.bbox_xyxy)
            containment = mask_intersection_over_smaller(r.mask, q.mask)

            if iou >= cfg.duplicate_iou_threshold:
                duplicate = True
                r.is_filtered = True
                r.filter_reason = f"duplicate_iou_{iou:.3f}"
                break

            if containment >= cfg.containment_threshold:
                duplicate = True
                r.is_filtered = True
                r.filter_reason = f"contained_{containment:.3f}"
                break

        if not duplicate:
            deduped.append(r)

    deduped = deduped[: cfg.max_regions_after_filter]
    return deduped


# ============================================================
# Semantic removal ratio-driven selection
# ============================================================

def select_regions_by_sr(
    regions: List[Region],
    sr: float,
    cfg: AttackConfig,
) -> Tuple[List[Region], float]:
    """
    Semantics:
    - sr = 0.0 -> no attack
    - sr = 1.0 -> remove all surviving semantic regions
    - 0 < sr < 1 -> select the lowest-priority prefix whose cumulative
      normalized semantic weight is closest to sr

    Returns the selected regions and the effective semantic removal ratio.
    """
    if cfg.selection_mode != "semantic_mass":
        raise ValueError(
            f"Unsupported selection mode: {cfg.selection_mode}. "
            "Expected 'semantic_mass'."
        )

    sr = float(sr)
    if not 0.0 <= sr <= 1.0:
        raise ValueError("sr must be in the interval [0, 1].")
    if not regions:
        return [], 0.0

    ordered = sorted(
        regions,
        key=lambda r: (r.priority_score, r.region_id),
    )

    semantic_weights = np.asarray(
        [max(0.0, r.semantic_weight) for r in ordered],
        dtype=np.float64,
    )
    weight_sum = float(semantic_weights.sum())
    if weight_sum <= 1e-12:
        semantic_weights = np.full(len(ordered), 1.0 / len(ordered), dtype=np.float64)
    else:
        semantic_weights /= weight_sum

    for region, weight in zip(ordered, semantic_weights):
        region.semantic_weight = float(weight)

    if sr == 0.0:
        return [], 0.0
    if sr == 1.0:
        return ordered, 1.0

    cumulative = np.concatenate(([0.0], np.cumsum(semantic_weights)))
    distances = np.abs(cumulative - sr)
    min_distance = float(distances.min())
    candidates = np.flatnonzero(np.isclose(distances, min_distance, rtol=0.0, atol=1e-12))
    k = int(candidates[0])

    return ordered[:k], float(cumulative[k])


# Fixed psychovisual experiment settings.
SR_VALUES = (0.30, 0.45, 0.60, 0.75, 0.90)
CONFIGURATION_NAME = "B"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".ppm", ".pgm"}


def make_attack_mask(image: Image.Image, regions: List[Region], sr: float,
                     cfg: AttackConfig) -> Dict[str, Any]:
    selected, achieved = select_regions_by_sr(regions, sr, cfg)
    union = union_masks([r.mask for r in selected], (image.height, image.width))
    refined = dilate_and_blur_mask(union, cfg.dilate_radius, cfg.blur_radius)
    binary = np.asarray(refined) > cfg.lama_binarize_threshold
    return {
        "selected": selected,
        "achieved_sr": achieved,
        "raw_mask_coverage_ratio": float(union.mean()),
        "final_mask_coverage_ratio": float(binary.mean()),
        "mask": np_mask_to_pil(binary),
        "mask_key": hashlib.sha256(np.packbits(binary).tobytes()).hexdigest(),
        "empty": not bool(union.any()),
    }


def resolve_device(requested: str) -> str:
    import torch
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; using CPU.", flush=True)
        return "cpu"
    return str(device)


def discover_images(input_dir: Path, output_dir: Path, recursive: bool) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different.")
    if output_dir in input_dir.parents:
        raise ValueError("The output directory must not be an ancestor of the input directory.")
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if output_dir == path.resolve().parent or output_dir in path.resolve().parents:
            continue
        images.append(path)
    images.sort(key=lambda p: (p.relative_to(input_dir).as_posix().casefold(), p.as_posix()))
    if not images:
        raise ValueError(f"No supported images found in {input_dir}")
    return images


def output_stems(images: List[Path], input_dir: Path) -> Dict[Path, str]:
    # Different formats/subfolders may contain the same stem; never overwrite them.
    counts = Counter(path.stem.casefold() for path in images)
    used = set()
    stems = {}
    for path in images:
        stem = path.stem[:160]
        if counts[path.stem.casefold()] > 1:
            digest = hashlib.sha256(path.relative_to(input_dir).as_posix().encode("utf-8")).hexdigest()[:10]
            stem += "_" + digest
        candidate, suffix = stem, 1
        while candidate.casefold() in used:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        used.add(candidate.casefold())
        stems[path] = candidate
    return stems


def run_batch(args: argparse.Namespace, *, mask_backend=None, inpaint_backend=None) -> int:
    """Generate five B outputs per input; optional backends support offline checks."""
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images = discover_images(input_dir, output_dir, args.recursive)
    stems = output_stems(images, input_dir)
    targets = [output_dir / f"{stems[path]}_{format_sr_tag(sr)}{CONFIGURATION_NAME}.png"
               for path in images for sr in SR_VALUES]
    if not args.overwrite:
        existing = next((path for path in targets if path.exists()), None)
        if existing is not None:
            raise FileExistsError(
                f"Output already exists: {existing}. Use a fresh directory or --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = AttackConfig(resize_max_side=args.resize_max_side)
    print(f"[INFO] {len(images)} images x {len(SR_VALUES)} ratios = {len(targets)} PNG outputs", flush=True)
    print(f"[INFO] Output directory: {output_dir}", flush=True)
    print(f"[INFO] Configuration B: all four coefficients = 0.25; sr = {SR_VALUES}", flush=True)
    if args.mask_backend == "dummy" or args.inpaint_backend == "dummy":
        print("[WARN] Dummy backend selected: diagnostic run, not a SAM2/LaMa experiment.", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    needs_torch = ((mask_backend is None and args.mask_backend == "sam2")
                   or (inpaint_backend is None and args.inpaint_backend == "lama"))
    if needs_torch:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is missing from the active Python environment.") from exc
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        args.device = resolve_device(args.device)
    mask_backend = mask_backend if mask_backend is not None else build_mask_backend(args)
    inpaint_backend = inpaint_backend if inpaint_backend is not None else build_inpaint_backend(args, cfg)
    # Initialize once and fail immediately on missing model packages/checkpoints.
    for backend in (mask_backend, inpaint_backend):
        if hasattr(backend, "_lazy_init"):
            backend._lazy_init()

    successes = failures = 0
    for image_index, path in enumerate(images, start=1):
        relative_name = path.relative_to(input_dir).as_posix()
        print(f"\n[IMAGE {image_index}/{len(images)}] {relative_name}", flush=True)
        try:
            image = load_image(str(path), max_side=cfg.resize_max_side)
            raw = mask_backend.generate(image)
            all_regions = build_regions_from_raw_masks(raw, image, cfg)
            retained = filter_regions(all_regions, cfg)
            regions = score_regions(retained, cfg)
            del raw, all_regions
            print(f"[INFO] Size: {image.width}x{image.height}; retained regions: {len(regions)}", flush=True)
            if not regions:
                print("[WARN] No retained regions: all five outputs will copy the input.", flush=True)
        except Exception as exc:
            failures += len(SR_VALUES)
            print(f"[ERROR] Cannot prepare {relative_name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        # Keep only paths in the cache to avoid retaining reconstructed images.
        cache = {}
        for sr in SR_VALUES:
            output_path = output_dir / f"{stems[path]}_{format_sr_tag(sr)}{CONFIGURATION_NAME}.png"
            print(f"[RUN] {relative_name}: B, sr={sr:.2f}", flush=True)
            try:
                mask_info = make_attack_mask(image, regions, sr, cfg)
                key = (mask_info["empty"], mask_info["mask_key"])
                cached_path = cache.get(key)
                if cached_path is None:
                    if mask_info["empty"]:
                        attacked = image.copy()
                    else:
                        attacked = inpaint_backend.inpaint(image, mask_info["mask"])
                    attacked = attacked.convert("RGB")
                    if attacked.size != image.size:
                        raise ValueError(f"Inpainting changed image size: {image.size} -> {attacked.size}")
                    save_image(attacked, str(output_path))
                    cache[key] = output_path
                else:
                    shutil.copyfile(cached_path, output_path)
                successes += 1
                detail = f" | reused {cached_path.name}" if cached_path is not None else ""
                print(f"[OK] {output_path.name} | effective sr={mask_info['achieved_sr']:.6f}"
                      f" | selected regions={len(mask_info['selected'])}{detail}", flush=True)
            except Exception as exc:
                failures += 1
                print(f"[ERROR] {output_path.name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n[DONE] Saved images: {successes}; failed conditions: {failures}", flush=True)
    print(f"[DONE] Output directory: {output_dir}", flush=True)
    return 1 if failures else 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAGA: five attacked PNGs per image, B coefficients fixed at 0.25; sr=0.30/0.45/0.60/0.75/0.90."
    )
    parser.add_argument("--input_dir", "--input", dest="input_dir", required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory for the five attacked PNGs per input")
    parser.add_argument("--mask_backend", default="sam2", choices=["dummy", "sam2"])
    parser.add_argument("--inpaint_backend", default="lama", choices=["dummy", "lama"])
    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2_ckpt", default="sam2_repo/checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--device", default="cpu", help="Device for SAM2 and LaMa (default: cpu)")
    parser.add_argument("--resize_max_side", type=int, default=1024,
                        help="Longest working side; 0 disables resizing (default: 1024)")
    parser.add_argument("--recursive", action="store_true", help="Also read input subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing named output images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.resize_max_side < 0:
        parser.error("--resize_max_side must be >= 0.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32).")
    return args


def build_mask_backend(args: argparse.Namespace) -> AutoMaskGeneratorBackend:
    if args.mask_backend == "dummy":
        return DummyGridMaskGenerator()
    return SAM2AutoMaskGeneratorBackend(args.sam2_cfg, args.sam2_ckpt, device=args.device)


def build_inpaint_backend(args: argparse.Namespace, cfg: AttackConfig) -> InpaintingBackend:
    if args.inpaint_backend == "dummy":
        return DummyInpaintingBackend()
    return SimpleLaMaBackend(binarize_threshold=cfg.lama_binarize_threshold, device=args.device)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run_batch(args)
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted. Completed images remain in the output directory.", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
