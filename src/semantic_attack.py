#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


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
    weight_area: float = 0.20
    weight_centrality: float = 0.40
    weight_foreground: float = 0.30
    weight_stability: float = 0.10

    # region selection
    selection_mode: str = "small_to_large"

    # mask refinement
    dilate_radius: int = 11
    blur_radius: int = 1
    lama_binarize_threshold: int = 16

    # performance / image resize
    resize_max_side: int = 1024

    # output
    save_region_masks: bool = False
    save_debug_overlay: bool = True
    save_union_mask: bool = True
    save_json: bool = True
    save_binary_union_mask: bool = True


# ============================================================
# I/O helpers
# ============================================================

def load_image(path: str, max_side: int = 1024) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
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
    return f"i{sr:.3f}"


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
    def __init__(self, binarize_threshold: int = 16) -> None:
        self._model = None
        self.binarize_threshold = binarize_threshold

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        try:
            from simple_lama_inpainting import SimpleLama
        except Exception as e:
            raise RuntimeError(
                "Could not import simple_lama_inpainting. Please install it in the active environment."
            ) from e

        print("[INFO] Initializing SimpleLaMa...", flush=True)
        self._model = SimpleLama()

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

    area_norm = normalize_scores([t["area_ratio"] for t in temp])
    cent_norm = normalize_scores([t["centrality_score"] for t in temp])
    fore_norm = normalize_scores([t["foreground_score"] for t in temp])
    stab_norm = normalize_scores([t["stability_score"] for t in temp])

    regions: List[Region] = []
    for t, a, c, f, s in zip(temp, area_norm, cent_norm, fore_norm, stab_norm):
        priority = (
            cfg.weight_area * a +
            cfg.weight_centrality * c +
            cfg.weight_foreground * f +
            cfg.weight_stability * s
        )
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
                priority_score=float(priority),
            )
        )

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

    kept.sort(key=lambda x: (x.stability_score, x.priority_score), reverse=True)

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
) -> List[Region]:
    """
    Semantics:
    - sr = 0.0 -> no attack
    - sr = 1.0 -> remove all surviving regions
    - increasing sr removes regions from smallest to largest
    """
    sr = float(max(0.0, min(1.0, sr)))
    if sr <= 0.0 or not regions:
        return []

    ordered = sorted(regions, key=lambda r: (r.area_ratio, r.priority_score))
    n = len(ordered)
    k = int(round(sr * n))
    k = max(0, min(k, n))
    return ordered[:k]


# ============================================================
# Debug visualization
# ============================================================

def make_overlay_image(
    image: Image.Image,
    regions: List[Region],
    selected_ids: Optional[set] = None,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    selected_ids = selected_ids or set()

    for r in regions:
        x1, y1, x2, y2 = r.bbox_xyxy
        outline = (255, 0, 0, 220) if r.region_id in selected_ids else (0, 180, 255, 180)
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=3)
        label = f"id={r.region_id} a={r.area_ratio:.3f}"
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=outline)

    return Image.alpha_composite(base, overlay).convert("RGB")


# ============================================================
# Attack pipeline
# ============================================================

class SemanticAutomaticAttack:
    def __init__(
        self,
        mask_backend: AutoMaskGeneratorBackend,
        inpaint_backend: InpaintingBackend,
        cfg: Optional[AttackConfig] = None,
    ) -> None:
        self.mask_backend = mask_backend
        self.inpaint_backend = inpaint_backend
        self.cfg = cfg or AttackConfig()

    def run(
        self,
        image: Image.Image,
        sr: float,
    ) -> Dict[str, Any]:
        print("[INFO] generating masks...", flush=True)
        raw_masks = self.mask_backend.generate(image)
        print(f"[INFO] raw masks count: {len(raw_masks)}", flush=True)

        print("[INFO] building regions...", flush=True)
        all_regions = build_regions_from_raw_masks(raw_masks, image, self.cfg)

        print("[INFO] filtering regions...", flush=True)
        filtered_regions = filter_regions(all_regions, self.cfg)
        print(f"[INFO] kept regions: {len(filtered_regions)}", flush=True)

        print("[INFO] selecting regions...", flush=True)
        selected_regions = select_regions_by_sr(filtered_regions, sr, self.cfg)
        print(f"[INFO] selected regions: {len(selected_regions)}", flush=True)

        selected_masks = [r.mask for r in selected_regions]
        union = union_masks(selected_masks, shape=(image.height, image.width))

        print("[INFO] refining union mask...", flush=True)
        refined_union_pil = dilate_and_blur_mask(
            union,
            dilate_radius=self.cfg.dilate_radius,
            blur_radius=self.cfg.blur_radius,
        )

        binary_union = (np.array(refined_union_pil) > self.cfg.lama_binarize_threshold).astype(np.uint8) * 255
        binary_union_pil = Image.fromarray(binary_union, mode="L")

        if union.sum() == 0:
            print("[INFO] empty union mask; copying original image", flush=True)
            attacked = image.copy()
        else:
            print("[INFO] inpainting...", flush=True)
            attacked = self.inpaint_backend.inpaint(image, refined_union_pil)
            print("[INFO] inpainting done", flush=True)

        return {
            "image": image,
            "attacked": attacked,
            "raw_masks_count": len(raw_masks),
            "regions_all": all_regions,
            "regions_filtered": filtered_regions,
            "regions_selected": selected_regions,
            "union_mask_bool": union,
            "union_mask_pil": refined_union_pil,
            "binary_union_mask_pil": binary_union_pil,
        }


# ============================================================
# Export
# ============================================================

def export_results(
    result: Dict[str, Any],
    output_dir: str,
    base_name: str,
    sr: float,
    cfg: AttackConfig,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    attacked = result["attacked"]
    union_mask_pil = result["union_mask_pil"]
    binary_union_mask_pil = result["binary_union_mask_pil"]
    filtered_regions = result["regions_filtered"]
    selected_regions = result["regions_selected"]
    image = result["image"]

    sr_tag = format_sr_tag(sr)
    stem = f"{base_name}_{sr_tag}"

    attacked_path = os.path.join(output_dir, f"{stem}_attacked.jpg")
    save_image(attacked, attacked_path)
    print(f"[INFO] saved attacked image: {attacked_path}", flush=True)

    if cfg.save_union_mask:
        mask_path = os.path.join(output_dir, f"{stem}_mask.png")
        save_image(union_mask_pil, mask_path)
        print(f"[INFO] saved union mask: {mask_path}", flush=True)

    if cfg.save_binary_union_mask:
        binary_mask_path = os.path.join(output_dir, f"{stem}_mask_binary.png")
        save_image(binary_union_mask_pil, binary_mask_path)
        print(f"[INFO] saved binary union mask: {binary_mask_path}", flush=True)

    if cfg.save_debug_overlay:
        sel_ids = {r.region_id for r in selected_regions}
        overlay = make_overlay_image(image, filtered_regions, selected_ids=sel_ids)
        overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
        save_image(overlay, overlay_path)
        print(f"[INFO] saved overlay: {overlay_path}", flush=True)

    if cfg.save_region_masks:
        masks_dir = os.path.join(output_dir, f"{stem}_regions")
        os.makedirs(masks_dir, exist_ok=True)
        for r in filtered_regions:
            m = np_mask_to_pil(r.mask)
            save_image(m, os.path.join(masks_dir, f"region_{r.region_id:03d}.png"))

    if cfg.save_json:
        json_path = os.path.join(output_dir, f"{stem}_metadata.json")
        payload = {
            "sr": sr,
            "raw_masks_count": result["raw_masks_count"],
            "regions_filtered": [
                {
                    **asdict(r),
                    "mask": None,
                }
                for r in filtered_regions
            ],
            "regions_selected_ids": [r.region_id for r in selected_regions],
            "selected_region_areas": [r.area_ratio for r in selected_regions],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[INFO] saved metadata: {json_path}", flush=True)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic region removal attack controlled by sr (semantic removal ratio)."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--sr",
        type=float,
        required=True,
        help="Semantic removal ratio in [0,1]",
    )

    parser.add_argument(
        "--mask_backend",
        type=str,
        default="sam2",
        choices=["dummy", "sam2"],
        help="Automatic mask generation backend (default: sam2)",
    )
    parser.add_argument(
        "--inpaint_backend",
        type=str,
        default="lama",
        choices=["dummy", "lama"],
        help="Prompt-free inpainting backend (default: lama)",
    )

    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 config (default: small)",
    )
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default="sam2_repo/checkpoints/sam2.1_hiera_small.pt",
        help="SAM2 checkpoint (default: small)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )

    return parser.parse_args()


def build_mask_backend(args: argparse.Namespace) -> AutoMaskGeneratorBackend:
    if args.mask_backend == "dummy":
        return DummyGridMaskGenerator()
    if args.mask_backend == "sam2":
        return SAM2AutoMaskGeneratorBackend(
            model_cfg=args.sam2_cfg,
            checkpoint=args.sam2_ckpt,
            device=args.device,
        )
    raise ValueError(f"Unsupported mask backend: {args.mask_backend}")


def build_inpaint_backend(args: argparse.Namespace, cfg: AttackConfig) -> InpaintingBackend:
    if args.inpaint_backend == "dummy":
        return DummyInpaintingBackend()
    if args.inpaint_backend == "lama":
        return SimpleLaMaBackend(binarize_threshold=cfg.lama_binarize_threshold)
    raise ValueError(f"Unsupported inpaint backend: {args.inpaint_backend}")


def main() -> None:
    args = parse_args()
    cfg = AttackConfig()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] output dir ready: {args.output_dir}", flush=True)

    image = load_image(args.input, max_side=cfg.resize_max_side)
    print(f"[INFO] loaded image: {args.input}", flush=True)
    print(f"[INFO] resized image size: {image.size}", flush=True)

    mask_backend = build_mask_backend(args)
    inpaint_backend = build_inpaint_backend(args, cfg)

    attack = SemanticAutomaticAttack(
        mask_backend=mask_backend,
        inpaint_backend=inpaint_backend,
        cfg=cfg,
    )

    print("[INFO] starting attack", flush=True)
    result = attack.run(image=image, sr=args.sr)

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    print("[INFO] exporting results...", flush=True)
    export_results(result, args.output_dir, base_name, args.sr, cfg)

    print("Done.", flush=True)
    print(f"Input                 : {args.input}", flush=True)
    print(f"Semantic removal (sr) : {args.sr:.3f}", flush=True)
    print(f"Output dir            : {args.output_dir}", flush=True)
    print(f"Raw masks             : {result['raw_masks_count']}", flush=True)
    print(f"Kept regions          : {len(result['regions_filtered'])}", flush=True)
    print(f"Selected regions      : {len(result['regions_selected'])}", flush=True)


if __name__ == "__main__":
    main()
