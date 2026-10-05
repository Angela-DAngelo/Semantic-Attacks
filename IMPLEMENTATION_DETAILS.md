# SAGA implementation details

**Table S1. SAGA implementation details.** This table documents the defaults and behavior of `semantic_attack_new.py`, using the SAM2 and LaMa backends. Regions are selected from the least to the most significant according to their priority scores. Command-line overrides of the checkpoint, device, input, output directory, and target `sr` must be recorded for each experiment.

| Stage | Parameter or operation | Value or behavior |
| --- | --- | --- |
| Input | Color mode | Convert the input image to RGB. |
| Input | `resize_max_side` | `1024` pixels; downsample only when the longest side exceeds this value. Preserve aspect ratio and round output dimensions to integer pixels. |
| Input | Resampling | Pillow Lanczos. All subsequent mask areas and coordinates refer to the image after this preprocessing. |
| SAM2 | Default backbone | SAM 2.1 Hiera Small. |
| SAM2 | Default configuration | `configs/sam2.1/sam2.1_hiera_s.yaml` |
| SAM2 | Default checkpoint | `sam2_repo/checkpoints/sam2.1_hiera_small.pt` |
| SAM2 | Backbone override | `--sam2_cfg` and `--sam2_ckpt`; the generator parameters below are unchanged when a different backbone is selected. |
| SAM2 | `--device` | Default: `cpu`. If exactly `cuda` is requested and CUDA is unavailable, fall back to CPU. This argument is passed to SAM2 only. |
| SAM2 | `points_per_side` | `12` |
| SAM2 | `points_per_batch` | `24` |
| SAM2 | `pred_iou_thresh` | `0.72` |
| SAM2 | `stability_score_thresh` | `0.90` |
| SAM2 | `stability_score_offset` | `1.0` |
| SAM2 | `crop_n_layers` | `0` |
| SAM2 | `box_nms_thresh` | `0.60` |
| SAM2 | `crop_n_points_downscale_factor` | `2`; no crop layers are requested in this configuration. |
| SAM2 | `min_mask_region_area` | `400` pixels, passed to the SAM2 generator. Optional SAM2 mask post-processing depends on the installed extension and package version. |
| SAM2 | `use_m2m` | `False` |
| Region attributes | Mask area $a_i$ | Number of foreground pixels divided by $HW$. |
| Region attributes | Centroid $(x_i,y_i)$ | Mean column and row coordinates of foreground pixels in the mask. |
| Region attributes | Bounding-box area $b_i$ | Inclusive axis-aligned box area divided by $HW$: `(x_max - x_min + 1) * (y_max - y_min + 1) / (H * W)` for a nonempty mask. |
| Region attributes | Centrality $c_i$ | $\max\{0,1-\sqrt{d_{x,i}^2+d_{y,i}^2}/\sqrt{2}\}$, where $d_{x,i}=(x_i-W/2)/(W/2)$ and $d_{y,i}=(y_i-H/2)/(H/2)$. Computed before feature normalization. |
| Region attributes | Foreground proxy $f_i$ | $0.55c_i+0.25\exp[-(a_i-0.10)^2/(2\cdot0.12^2)]+0.20\exp[-(b_i-0.18)^2/(2\cdot0.18^2)]$. Uses the centrality value before feature normalization. The function clips its result to $[0,1]$. |
| Region attributes | Meaning of $f_i$ | A geometric heuristic based on centrality, mask area, and bounding-box area; no additional learned saliency or foreground model is used. |
| Region attributes | Stability $s_i$ | SAM2 `stability_score`; the SAM2 adapter uses `0.5` if this field is absent. |
| Filtering | `min_area_ratio` | `0.005`; discard regions with smaller area ratios. |
| Filtering | `max_area_ratio` | `1/3`; discard regions with larger area ratios. |
| Filtering | Deduplication order | Process candidates by descending raw stability score, then descending mask area ratio. Equal keys preserve the incoming order. |
| Filtering | `duplicate_iou_threshold` | `0.60`; discard a candidate if its **bounding-box IoU** with any already retained region is at least this value. This is not mask IoU. |
| Filtering | `containment_threshold` | `0.90`; discard a candidate if its mask intersection with an already retained mask, divided by the smaller mask area, is at least this value. |
| Filtering | `max_regions_after_filter` | Retain the first `64` regions after deduplication, in the order defined above. |
| Scoring | Normalization order | Normalize features and compute priorities after filtering and truncation to the retained set. |
| Scoring | Feature normalization | Min–max normalization across retained regions for $c_i,f_i,s_i$. The area ratio $a_i$ is not min–max normalized. |
| Scoring | Constant-feature fallback | If `abs(max - min) < 1e-12`, assign normalized value `1.0` to every retained region for that feature. |
| Scoring | Priority formula | $p_i=\lambda_a a_i+\lambda_c c_i+\lambda_f f_i+\lambda_s s_i$, using normalized $c_i,f_i,s_i$. |
| Scoring | Priority coefficients | `weight_area = 0.20`, `weight_centrality = 0.40`, `weight_foreground = 0.30`, `weight_stability = 0.10`. |
| Scoring | Coefficient validation | Coefficients must be nonnegative and sum to one within absolute tolerance `1e-9`. They are configured in `AttackConfig`, not exposed as command-line arguments. |
| Selection | Semantic weights | $w_i=p_i/\sum_jp_j$. If the sum of nonnegative scores is at most `1e-12`, use uniform weights. The selection function renormalizes the weights to sum to one. |
| Selection | Removal order | Ascending priority score, then ascending region ID. Region IDs are indices of the original mask proposals. |
| Selection | `selection_mode` | `semantic_mass`; select a prefix of the ascending ordering whose cumulative weight is closest to the target `sr`, including the empty prefix. |
| Selection | Target validation | `sr` must lie in $[0,1]$. |
| Selection | Equal-distance rule | Choose the smaller prefix length. Distances are considered equal with absolute tolerance `1e-12` and zero relative tolerance. |
| Selection | Endpoints | `sr = 0`: select no regions. `sr = 1`: select all retained regions when the retained set is nonempty. |
| Selection | No retained regions | Select no regions and return `achieved_sr = 0.0` for every valid target, including `sr = 1`. |
| Selection | Effective removal | `achieved_sr` is the sum of the selected normalized region weights. It measures assigned semantic mass, not pixel coverage or a measured loss of perceptual identity. |
| Mask | Union | Pixelwise Boolean union of the selected region masks. |
| Mask | `dilate_radius` | `11` pixels; Pillow `MaxFilter(size=23)`, corresponding to a square $23\times23$ structuring element on the binary mask. |
| Mask | `blur_radius` | Pillow `GaussianBlur(radius=1)`, applied after dilation. |
| Mask | `lama_binarize_threshold` | `16`; values strictly greater than `16` become `255`, all others become `0`. The threshold is applied on the 0–255 scale. |
| Mask | `mask_coverage_ratio` | Mean of the Boolean union **before** dilation, smoothing, and thresholding. It is not the area fraction of the final LaMa mask. |
| Reconstruction | Backend | `simple_lama_inpainting.SimpleLama()`, with no constructor arguments. Record the wrapper version and actual model checkpoint used by the installed package. |
| Reconstruction | Device | No explicit LaMa device is passed by this script; `--device` configures SAM2 only. |
| Reconstruction | Mask supplied to LaMa | The refined mask after thresholding, as a binary grayscale image. |
| Reconstruction | Empty union | Skip LaMa and copy the preprocessed RGB input image. |
| Export | Attacked image | `{input_stem}_i{sr:.3f}_attacked.jpg`; JPEG is written through Pillow without explicit quality or subsampling arguments. |
| Export | Refined mask | `{input_stem}_i{sr:.3f}_mask.png`; mask after dilation and smoothing, before thresholding. |
| Export | Binary mask | `{input_stem}_i{sr:.3f}_mask_binary.png`; thresholded mask supplied to LaMa. |
| Export | Debug overlay | Enabled by default. Bounding boxes are red for selected regions and blue for other retained regions. Labels contain `id`, priority `p`, and semantic weight `w`; the boxes are not the inpainting masks. |
| Export | Per-region masks | Disabled by default: `save_region_masks = False`. |
| Export | Metadata | Enabled by default. JSON records target and effective `sr`, union coverage, priority coefficients, retained-region attributes, and selected-region IDs, areas, scores, and weights. |
| Export | Timing | Export starts after the reconstruction stage returns. |

The foreground proxy and centrality expressions above describe the attributes before min–max normalization. The same mathematical symbols are reused for their normalized values in the priority formula, as in Section 5.2 of the paper.

The table records the current implementation, including its image preprocessing and JPEG export. Consequently, `sr = 0` skips region removal and inpainting but does not guarantee that the exported pixels match the original input file. To isolate region removal from resizing and JPEG compression, the experimental protocol must control these operations explicitly.

For reproducibility, archive the script revision, input images, complete command lines, Python and package versions, SAM2 repository revision, checkpoint identifiers or hashes, and the actual device used by each model. These environment-specific values are not fixed by the script. In the CPU execution reported during development, SAM2 warned that the optional `_C` extension was unavailable and that its associated mask post-processing was skipped; record whether this step is enabled for every reported experiment.
