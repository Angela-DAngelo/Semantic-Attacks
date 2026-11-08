# SAPIRE: Semantic-Aware Perceptual Identity Region Editing

SAPIRE is a prompt-free image transformation pipeline for investigating watermark robustness in relation to perceptual identity. It extracts candidate regions with SAM2, ranks the retained regions using geometric and segmentation attributes, and reconstructs selected regions with LaMa.

This implementation provides **configuration B**, with equal weights for area, centrality, foreground relevance, and segmentation stability. The executable interfaces use five semantic removal ratios:

```text
sr = 0.30, 0.45, 0.60, 0.75, 0.90
```

The parameter `sr` is a target fraction of normalized region-priority mass. It is not the fraction of image pixels edited, a watermark-removal success rate, or a universal measure of perceptual identity loss.

## 1. Scripts and their roles

| File | Role | Input | Output |
| --- | --- | --- | --- |
| `sapire_core.py` | Implements the pipeline and provides a standalone batch command. | A directory of images. | Five RGB PNGs per successfully processed image. |
| `sapire.py` | Adapts the core to the WIBE attack interface. | One RGB floating-point tensor with shape `(3, H, W)`. | One transformed CPU `float32` tensor with the same shape. |

The adapter was supplied as `sapire(1).py`; install it as `sapire.py`. The separately supplied `semantic_attack_B_5sr(2).py` is byte-for-byte identical to `sapire_core.py` and is not needed as an additional file.

The adapter imports `sapire_core.py` from the configured project directory. Keep that filename unchanged. The core can also be run directly; WIBE is not required for standalone image generation.

Some help text in the supplied core still uses the earlier label `SAGA` or the filename `semantic_attack_B_5sr.py`. These refer to the same supplied implementation.

## 2. Environment and model files

### Dependencies

| Component | Required for |
| --- | --- |
| NumPy and Pillow | Image handling, masks, geometry, and both execution modes. |
| PyTorch | SAM2, LaMa, and the WIBE tensor interface. |
| An installed SAM2 package exposing `sam2.build_sam` and `sam2.automatic_mask_generator` | Real segmentation. |
| `simple-lama-inpainting`, exposing `simple_lama_inpainting.SimpleLama` | Real inpainting. |
| WIBE, exposing `wibench.common.attacks.BaseAttack` | The adapter, including its smoke test. |

LPIPS, DreamSim, and Excel-writing packages are not required by the standalone core. They may be needed by separate evaluation scripts.

Use the environment and repository revisions used for the experiments. The supplied scripts do not specify exact dependency versions. A reproducible release should include an environment specification or lockfile from the experimental machine, together with the SAM2 and WIBE commit identifiers. CPU and CUDA execution are supported; the installed PyTorch build must support the selected device.

### Expected paths

The examples below assume that commands are run from the project root. For WIBE integration, use the layout expected by the supplied adapter:

| Path relative to the project root | Contents |
| --- | --- |
| `sapire_core.py` | The unchanged core module. |
| `sam2_repo/checkpoints/sam2.1_hiera_small.pt` | SAM2.1 Hiera Small checkpoint. |
| `models/big-lama.pt` | LaMa checkpoint selected by the adapter. |
| `wibe/` | The installed WIBE checkout. |
| `wibe/user_plugins/sapire.py` | The WIBE adapter. |

The SAM2 configuration identifier is `configs/sam2.1/sam2.1_hiera_s.yaml`. It must be resolvable by the installed SAM2 configuration system.

Obtain the checkpoints required by the installed backends and record their sources and SHA-256 hashes in the experimental release. The adapter checks that its two checkpoint files exist and are non-empty; it does not verify their hashes.

The adapter sets `LAMA_MODEL` to the local `models/big-lama.pt` during LaMa initialization and inference. The standalone command has no `--lama_ckpt` option: configure the installed SimpleLama loader to select the intended checkpoint. The standalone examples use the same `LAMA_MODEL` convention as the adapter; the installed loader must support it.

## 3. Standalone batch generation

### Generate the five SAPIRE-B outputs

From the project root, with the environment active and checkpoints available:

```bash
LAMA_MODEL="$PWD/models/big-lama.pt" python sapire_core.py \
  --input_dir images \
  --output_dir out_B \
  --mask_backend sam2 \
  --inpaint_backend lama \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_ckpt sam2_repo/checkpoints/sam2.1_hiera_small.pt \
  --device cuda \
  --seed 0
```

This environment-variable syntax is for a POSIX shell. On other shells, set `LAMA_MODEL` using that shell's syntax before running Python.

Use `--device cpu` for CPU execution. The standalone default is `cpu`. If CUDA is requested but unavailable, the standalone runner falls back to CPU and prints a warning.

For an input named `example.jpg`, successful processing creates:

```text
example_sr0.30B.png
example_sr0.45B.png
example_sr0.60B.png
example_sr0.75B.png
example_sr0.90B.png
```

Outputs are collected in the specified output directory. If input filenames have colliding stems, the runner assigns distinct output stems, including a relative-path hash where needed.

### Preprocessing and watermark evaluation

The standalone loader converts each image to RGB. By default, it downsamples images whose longest side exceeds 1024 pixels using Pillow Lanczos resampling. It preserves aspect ratio, rounds the output dimensions to integer pixels, and does not upscale smaller images.

To disable resizing:

```bash
LAMA_MODEL="$PWD/models/big-lama.pt" python sapire_core.py \
  --input_dir watermarked_images \
  --output_dir out_B \
  --resize_max_side 0 \
  --device cuda \
  --seed 0
```

For the protocol expected by the WIBE adapter, prepare the source images with longest side at most 1024 pixels **before watermark embedding**. Preserve those dimensions in the watermarked inputs. Applying an additional resize after embedding changes the evaluated transformation and must be recorded if used.

### Command-line options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input_dir`, alias `--input` | Required | Input image directory. |
| `--output_dir` | Required | Output directory. |
| `--mask_backend` | `sam2` | `sam2` for the real model; `dummy` for diagnostics. |
| `--inpaint_backend` | `lama` | `lama` for the real model; `dummy` for diagnostics. |
| `--sam2_cfg` | `configs/sam2.1/sam2.1_hiera_s.yaml` | SAM2 configuration identifier. |
| `--sam2_ckpt` | `sam2_repo/checkpoints/sam2.1_hiera_small.pt` | SAM2 checkpoint path. |
| `--device` | `cpu` | Device used by the real backends. |
| `--resize_max_side` | `1024` | Maximum working side; `0` disables resizing. |
| `--recursive` | Disabled | Search input subdirectories. |
| `--overwrite` | Disabled | Allow replacement of matching output filenames. |
| `--seed` | `0` | Integer seed in `[0, 2**32)`. |

There is no standalone `--sr` or coefficient option. The command generates all five fixed levels with configuration B. The underlying selection function accepts any `sr` in `[0, 1]` when called programmatically.

### Input discovery and failures

Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`, `.gif`, `.ppm`, and `.pgm`. Animated or multipage files use the first frame. Input discovery is non-recursive unless requested, and excludes the output subtree.

The input and output directories must differ, and the output directory cannot be an ancestor of the input directory. If a target output already exists, the runner stops before generation unless `--overwrite` is supplied. Existing outputs are not automatically treated as completed conditions for resuming a run.

The real backends initialize before image processing so missing model packages or checkpoints fail early. Later image- or condition-specific failures are printed, and processing continues where possible. The final summary reports saved images and failed conditions. The runner returns `0` on success, `1` on failure, and `130` on interruption.

### Diagnostic run without model inference

```bash
python sapire_core.py \
  --input_dir images \
  --output_dir out_dummy \
  --mask_backend dummy \
  --inpaint_backend dummy
```

This mode uses synthetic rectangular regions and a blur-based reconstruction. It exercises the batch workflow with NumPy and Pillow, without requiring SAM2, LaMa, or PyTorch. Its outputs are diagnostic images and must not be used as SAM2/LaMa experimental results.

## 4. Pipeline and implementation settings

### Region extraction

SAM2.1 Hiera Small extracts candidate binary masks from the working RGB image. The generator is constructed with these explicit arguments:

| SAM2 argument | Value |
| --- | --- |
| `points_per_side` | `12` |
| `points_per_batch` | `24` |
| `pred_iou_thresh` | `0.72` |
| `stability_score_thresh` | `0.90` |
| `stability_score_offset` | `1.0` |
| `crop_n_layers` | `0` |
| `box_nms_thresh` | `0.60` |
| `crop_n_points_downscale_factor` | `2` |
| `min_mask_region_area` | `400` |
| `use_m2m` | `False` |

Other SAM2 behavior follows the installed package version, which should be recorded with the experiment.

### Region filtering

For each candidate, the core computes its mask area, axis-aligned bounding box, mask centroid, centrality, foreground score, and stability. It then applies the following filtering rules:

| Operation | Setting or behavior |
| --- | --- |
| Minimum mask area ratio | Discard if below `0.005`. |
| Maximum mask area ratio | Discard if above `1/3`. |
| Candidate traversal | Descending stability, then descending area ratio. |
| Duplicate criterion | Discard if bounding-box IoU with an already retained region is at least `0.60`. |
| Containment criterion | Discard if mask intersection divided by the smaller mask area is at least `0.90`. |
| Maximum retained regions | Keep the first `64` regions after de-duplication. |

The duplicate IoU criterion uses **bounding boxes**; the containment criterion uses **binary masks**. The two criteria are distinct. Filtering precedes the normalization used for priority scoring.

### Region attributes and priority

For an image of width `W` and height `H`, the normalized mask area is:

```text
a_i = mask_area_i / (H * W)
```

For mask centroid `(x_i, y_i)`, raw centrality is:

```text
dx_i = (x_i - W/2) / (W/2)
dy_i = (y_i - H/2) / (H/2)
c_i = max(0, 1 - sqrt(dx_i**2 + dy_i**2) / sqrt(2))
```

Let `b_i` be the bounding-box area divided by `H * W`, counting both endpoint pixels. The foreground proxy uses raw centrality:

```text
f_i = 0.55 * c_i
    + 0.25 * exp(-(a_i - 0.10)**2 / (2 * 0.12**2))
    + 0.20 * exp(-(b_i - 0.18)**2 / (2 * 0.18**2))
```

This is a geometric heuristic. It does not use an additional learned foreground or saliency model.

Centrality, foreground score, and stability are independently min-max normalized over the retained regions. If an attribute's range is below `1e-12`, every normalized value for that attribute is set to `1`. The area term remains `mask_area / (H * W)`; it is not min-max normalized.

Configuration B assigns:

```text
weight_area = weight_centrality = weight_foreground = weight_stability = 0.25
p_i = 0.25 * (a_i + c_i_normalized + f_i_normalized + s_i_normalized)
```

Larger priorities indicate regions removed later. The core checks that the coefficients are non-negative and sum to one within numerical tolerance.

### Semantic mass and region selection

Priorities are normalized to semantic weights:

```text
w_i = p_i / sum_j(p_j)
```

If the priority sum is at most `1e-12`, uniform weights are used. Regions are ordered by ascending priority, with the original region identifier as a deterministic tie-breaker.

The cumulative masses include the empty prefix:

```text
C_0 = 0
C_k = sum of the first k ordered semantic weights
```

For `0 < sr < 1`, the selected regions are the prefix minimizing `abs(C_k - sr)`. If distances are equal within an absolute tolerance of `1e-12`, the smaller prefix is selected. `sr = 0` selects no regions, and `sr = 1` selects all retained regions. If no regions survive filtering, the selection is empty at every level.

The effective semantic removal ratio is the selected cumulative mass. Because whole regions are selected, it can differ from the requested `sr`; different target levels can also produce the same selection.

### Mask refinement and reconstruction

| Operation | Setting or behavior |
| --- | --- |
| Initial removal mask | Union of the selected binary masks. |
| Dilation | Pillow `MaxFilter` with radius parameter `11`, giving a `23 x 23` square window. |
| Smoothing | Pillow `GaussianBlur(radius=1)`. |
| Binarization | Select values strictly greater than `16` on the 8-bit mask. |
| Reconstruction | Prompt-free inpainting through `SimpleLaMaBackend`. |
| Output dimensions | Preserve input dimensions. If LaMa returns only right/bottom padding to multiples of eight, crop that padding; unexpected sizes raise an error. |
| Empty selection | Return a copy of the working input without inpainting. |

The raw union coverage, final mask coverage after refinement, and semantic mass are different quantities. Dilation and smoothing can expand the final edited support beyond the selected mask union.

Segmentation, filtering, and scoring run once per input in the standalone runner. Reconstructions are reused when final masks are identical, while all five named output files are still saved.

## 5. WIBE integration

### Adapter interface

```python
SAPIRE(sr=0.30, device="cuda", project_dir=None, seed=0)
```

| Argument | Behavior |
| --- | --- |
| `sr` | One of `0.30`, `0.45`, `0.60`, `0.75`, `0.90`. |
| `device` | `cpu` or a CUDA device such as `cuda` or `cuda:0`; default `cuda`. |
| `project_dir` | Root containing `sapire_core.py` and the expected checkpoints. Default: `~/sapire-wibe`. |
| `seed` | Integer in `[0, 2**32)`; default `0`. |

Each call accepts one finite floating-point RGB tensor in `[0, 1]` with shape `(3, H, W)` and longest side at most 1024 pixels. Batched inputs are not supported. The returned tensor is on the CPU, has dtype `float32`, and has the same shape and range. The adapter does not modify the supplied tensor.

The adapter performs no resizing and rejects larger inputs. CUDA requests fail if CUDA is unavailable; unlike the standalone runner, the adapter does not fall back to CPU.

### RGB8 conversion and baseline

Before segmentation, the adapter converts the input tensor to an RGB8 image using rounding:

```text
RGB8 = round(255 * clamp(input, 0, 1))
```

The reconstructed image is returned as `RGB8_output / 255`. An empty selection therefore returns the quantized input. For arbitrary floating-point inputs, this can introduce a small pixel change even when inpainting is skipped. Account for this conversion in the baseline, and document whether the benchmark uses saved 8-bit watermarked inputs or tensors with additional precision.

### Creating the five attack instances

With the adapter installed under `wibe/user_plugins/`, the following illustrates instance creation from the `wibe/` directory in the configured WIBE environment:

```python
from pathlib import Path
from user_plugins.sapire import SAPIRE

project_dir = Path.cwd().parent
attacks = {
    f"SAPIRE_B_sr{sr:.2f}": SAPIRE(
        sr=sr,
        device="cuda",
        project_dir=project_dir,
        seed=0,
    )
    for sr in (0.30, 0.45, 0.60, 0.75, 0.90)
}
```

Pass these instances to the experiment runner using the configuration mechanism of the pinned WIBE revision. Use a distinct `report_name` for each level. This Python example creates attacks; it does not launch embedding, detection, or metric computation. The adapter does not provide a complete WIBE experiment configuration.

### Real-model smoke test

From the project root:

```bash
python wibe/user_plugins/sapire.py \
  --smoke-test \
  --project-dir . \
  --device cuda
```

WIBE must already be importable. Use `--device cpu` when needed. The smoke test creates a synthetic image in memory, exercises all five levels, checks output shape, dtype, range, input preservation, and a cached repeat. It also runs real LaMa on a non-empty mask, even if the synthetic image produces no retained SAM2 regions. Successful completion prints `PLUGIN_SAPIRE_OK`.

The smoke test writes no dataset images or result files. It checks execution of the installed adapter and backends; it does not reproduce the paper's quantitative results.

### Caching, randomness, and errors

Model instances are shared within a worker process for the same project directory, device, and seed. The image cache retains only the most recent quantized image, identified by its shape and content hash. Regions and reconstructed outputs are reused for that image; a different quantized image invalidates the image cache. Grouping conditions for the same image can improve reuse. Start with one GPU worker when checking resource requirements.

The adapter seeds inference and restores Python, NumPy, and PyTorch random states, along with the cuDNN settings it changes. This isolates attack randomness from watermark and payload randomness. Exact outputs can still depend on backend versions, checkpoint files, device, and numerical behavior, which should be recorded.

The cache is in memory and does not resume an interrupted experiment. Saving benchmark outputs and experiment-level resume belong to the WIBE runner. The plugin itself writes no masks, image files, semantic reports, or metric tables. Exceptions propagate to WIBE; they are not converted into successful unchanged-image outputs.

## 6. Core integrity check

Before import, the adapter verifies the SHA-256 hash of `sapire_core.py`. The supplied core and its duplicate `semantic_attack_B_5sr(2).py` have this hash:

```text
4dbe7f5aacaa5f3cacb97039c13d9639b4fb5f0c2c29d923825f96d6454511f1
```

Changing the core's contents, including comments, whitespace, or line endings, changes its hash. After an intentional core revision, compute the new hash and update `CORE_SHA256` in the adapter so the published files remain a matched pair:

```bash
sha256sum sapire_core.py
```

Renaming an identical copy to `sapire_core.py` does not change its contents or hash. The integrity check identifies the core version; it does not pin dependencies, checkpoints, or experimental data.

## 7. Scope of reproducibility

These modules implement SAPIRE-B image generation and its WIBE interface. To reproduce the experiments reported in *Rethinking Robustness in Digital Watermarking: A Perceptual Identity-Based Framework*, accompany them with:

- Exact dependency versions, repository commits, checkpoint sources and hashes, execution device, seeds, and commands.
- COCO image identifiers and the actual subsets used for each study.
- The complete WIBE configuration, watermarking-model settings and checkpoints, payloads or their generation procedure, detection thresholds, and saved-input preprocessing.
- The ablation runner for configurations A-F at `sr = 0.40, 0.65, 0.90`; the supplied command-line interfaces expose only B and the five main levels.
- The configurations and execution instructions for the comparison attacks.
- The psychovisual response data or appropriate shareable aggregates, analysis scripts, and statistical settings needed to regenerate the reported tables and figures.

The standalone core exports transformed images and console summaries. BER, detection rates, PSNR, SSIM, LPIPS, DreamSim, psychovisual analyses, and paper figures require the corresponding external evaluation workflows.
