# SAPIRE implementation details

This document describes the SAM2 and LaMa implementation of Semantic-Aware Perceptual Identity Region Editing (SAPIRE) in `semantic_attack_new.py`. It distinguishes the script defaults from the experimental configurations reported in the manuscript.

SAPIRE orders retained regions from the least to the most significant according to their priority scores. The target `sr` controls the cumulative normalized semantic mass assigned to selected regions. It is not a target fraction of edited pixels, a measured loss of perceptual identity, or a calibrated perceptual-identity threshold.

The script defaults below use **configuration A**. The reference setting reported for the psychovisual studies and watermarking evaluation is **configuration B**, with four equal priority coefficients. When reproducing these experiments, set the coefficients explicitly in `AttackConfig`; changing `sr` alone does not select configuration B.

## Pipeline settings and behavior

**Table S1. SAPIRE implementation details.** The table documents the script defaults and numerical behavior. Experiment-specific overrides of the configuration, checkpoint, device, input, output directory, and target `sr` must be recorded together with the script revision.

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
| SAM2 | `min_mask_region_area` | `400` pixels, passed to the SAM2 generator. Whether the associated optional mask post-processing executes depends on the installed extension and package version; record its actual execution status. |
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
| Filtering | `duplicate_iou_threshold` | `0.60`; discard a candidate if its **bounding-box IoU** with any already retained region is at least this value. This criterion uses bounding boxes rather than mask IoU. |
| Filtering | `containment_threshold` | `0.90`; discard a candidate if its mask intersection with an already retained mask, divided by the smaller mask area, is at least this value. |
| Filtering | `max_regions_after_filter` | Retain the first `64` regions after deduplication, in the order defined above. |
| Scoring | Normalization order | Normalize features and compute priorities after filtering and truncation to the retained set. |
| Scoring | Feature normalization | Min-max normalization across retained regions for $c_i,f_i,s_i$. The area ratio $a_i$ is not min-max normalized. |
| Scoring | Constant-feature fallback | If `abs(max - min) < 1e-12`, assign normalized value `1.0` to every retained region for that feature. |
| Scoring | Priority formula | $p_i=\lambda_a a_i+\lambda_c c_i+\lambda_f f_i+\lambda_s s_i$, using normalized $c_i,f_i,s_i$. |
| Scoring | Script-default coefficients: configuration A | `weight_area = 0.20`, `weight_centrality = 0.40`, `weight_foreground = 0.30`, `weight_stability = 0.10`. |
| Scoring | Manuscript reference coefficients: configuration B | `weight_area = weight_centrality = weight_foreground = weight_stability = 0.25`. Set these values explicitly in `AttackConfig` when using the script defaults listed above. |
| Scoring | Coefficient validation and configuration | Coefficients must be nonnegative and sum to one within absolute tolerance `1e-9`. They are configured in `AttackConfig` and are not exposed as command-line arguments. Record the actual coefficients for every run. |
| Selection | Semantic weights | $w_i=p_i/\sum_jp_j$. If the sum of nonnegative scores is at most `1e-12`, use uniform weights. The selection function renormalizes the weights to sum to one. |
| Selection | Removal order | Ascending priority score, then ascending region ID. Region IDs are indices of the original mask proposals. |
| Selection | `selection_mode` | `semantic_mass`; select a prefix of the ascending ordering whose cumulative weight is closest to the target `sr`, including the empty prefix. |
| Selection | Target validation | `sr` must lie in $[0,1]$. |
| Selection | Equal-distance rule | Choose the smaller prefix length. Distances are considered equal with absolute tolerance `1e-12` and zero relative tolerance. |
| Selection | Endpoints | `sr = 0`: select no regions. `sr = 1`: select all retained regions when the retained set is nonempty. |
| Selection | No retained regions | Select no regions and return `achieved_sr = 0.0` for every valid target, including `sr = 1`. |
| Selection | Effective removal | `achieved_sr` is the sum of the selected normalized region weights. It measures assigned semantic mass rather than pixel coverage or measured perceptual-identity loss. |
| Mask | Union | Pixelwise Boolean union of the selected region masks. |
| Mask | `dilate_radius` | `11` pixels; Pillow `MaxFilter(size=23)`, corresponding to a square $23\times23$ structuring element on the binary mask. |
| Mask | `blur_radius` | Pillow `GaussianBlur(radius=1)`, applied after dilation. |
| Mask | `lama_binarize_threshold` | `16`; values strictly greater than `16` become `255`, all others become `0`. The threshold is applied on the 0-255 scale. |
| Mask | `mask_coverage_ratio` | Mean of the Boolean union **before** dilation, smoothing, and thresholding. It is not the area fraction of the final LaMa mask. |
| Reconstruction | Backend | `simple_lama_inpainting.SimpleLama()`, with no constructor arguments. Record the wrapper version and actual model checkpoint used by the installed package. |
| Reconstruction | Device | No explicit LaMa device is passed by this script; `--device` configures SAM2 only. Record the device actually used by LaMa separately. |
| Reconstruction | Mask supplied to LaMa | The refined mask after thresholding, as a binary grayscale image. |
| Reconstruction | Empty union | Skip LaMa and copy the preprocessed RGB input image. |
| Export | Attacked image | `{input_stem}_i{sr:.3f}_attacked.jpg`; JPEG is written through Pillow without explicit quality or subsampling arguments. |
| Export | Refined mask | `{input_stem}_i{sr:.3f}_mask.png`; mask after dilation and smoothing, before thresholding. |
| Export | Binary mask | `{input_stem}_i{sr:.3f}_mask_binary.png`; thresholded mask supplied to LaMa. |
| Export | Debug overlay | Enabled by default. Bounding boxes are red for selected regions and blue for other retained regions. Labels contain `id`, priority `p`, and semantic weight `w`; the boxes are visualization annotations rather than the inpainting masks. |
| Export | Per-region masks | Disabled by default: `save_region_masks = False`. |
| Export | Metadata | Enabled by default. JSON records target and effective `sr`, union coverage, priority coefficients, retained-region attributes, and selected-region IDs, areas, scores, and weights. Environment information not recorded automatically must be archived separately. |
| Export | Timing | Export starts after the reconstruction stage returns. |

The foreground proxy and centrality expressions in Table S1 describe the attributes before min-max normalization. The same mathematical symbols are reused for their normalized values in the priority formula, as in Section 5.2 of the manuscript.

For a fixed input and fixed pipeline settings, increasing `sr` selects progressively longer prefixes of the retained-region ordering. Selection is discrete: different targets can select the same regions, and `achieved_sr` can differ from the target. This ordering does not guarantee monotonic perceptual-identity degradation or a perceptual-identity boundary crossing for every image.

## Region-priority configurations

**Table S2. Coefficient configurations used in the ablation study.** The coefficient order is area, centrality, foreground proxy, and stability.

| Configuration | $\lambda_a$ | $\lambda_c$ | $\lambda_f$ | $\lambda_s$ | Description |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 0.20 | 0.40 | 0.30 | 0.10 | Initial heuristic setting and script default documented in Table S1. |
| B | 0.25 | 0.25 | 0.25 | 0.25 | Equal weighting; manuscript reference configuration. |
| C | 0 | 1/2 | 3/8 | 1/8 | Omit the direct area term from A and renormalize the remaining coefficients. |
| D | 1/3 | 0 | 1/2 | 1/6 | Omit the direct centrality term from A and renormalize the remaining coefficients. |
| E | 2/7 | 4/7 | 0 | 1/7 | Omit the foreground-proxy term from A and renormalize the remaining coefficients. |
| F | 2/9 | 4/9 | 1/3 | 0 | Omit the direct stability term from A and renormalize the remaining coefficients. |

Omitting a direct term does not eliminate all associated information. The foreground proxy incorporates area and centrality, while stability also participates in region filtering.

Configuration B was retained as an operational reference following the ablation study. This choice does not establish that B is uniquely superior or universally optimal.

## Experimental profiles reported in the manuscript

**Table S3. SAPIRE experimental profiles.** The reference segmentation and reconstruction backends are SAM2.1 Hiera Small and LaMa. Preprocessing, segmentation, filtering, mask refinement, and reconstruction settings are held fixed across conditions within each experiment.

| Experiment | Source images | Configuration | Target `sr` values | Number of conditions or judgments |
| --- | --- | --- | --- | --- |
| Ablation study, Section 5.3 | 100 COCO images | A-F | 0.40, 0.65, 0.90 | 1,800 original-transformed image-pair conditions. |
| Psychovisual study, Section 5.4 | 87 COCO images | B | 0.30, 0.45, 0.60, 0.75, 0.90 | 435 image pairs; 20 judgments per pair; 8,700 judgments. |
| Watermark benchmark, Section 6.1 | 100 COCO images; TrustMark Q, InvisMark, and Watermark Anything | B | 0.30, 0.45, 0.60, 0.75, 0.90 | 1,500 attacked outputs, in addition to the no-attack reference conditions. |
| Joint PI-detection study, Section 6.2 | 30 sources selected from the watermark benchmark; all three watermarking systems | B | 0.30, 0.45, 0.60, 0.75, 0.90 | 450 image pairs; 10 judgments per pair; 4,500 judgments. |

### Ablation study

For each source image, the same extracted and filtered regions are reused across all six coefficient configurations and all three removal levels. Feature normalization uses this fixed retained set; changing the coefficients changes priorities and selection without introducing a new segmentation.

The LPIPS and DreamSim trajectories are evaluated against the corresponding source image. The qualitative examples in Figure 6 use configuration B at `sr = 0.40, 0.65, 0.90` on selected images from the ablation subset.

### Psychovisual study

The 87-image psychovisual subset has no overlap with either the ablation subset or the watermark benchmark subset. Each of the 20 observers evaluates all 435 pairs once, in an independently randomized order, using the binary identity-preservation task described in Section 5.4.

The exact source images and transformed stimuli used in the study must be identified in the experiment records. Archive the evaluated stimuli or provide a linked archive so that the human judgments remain associated with the actual images presented.

### Watermark benchmark and joint PI-detection study

SAPIRE is applied to the watermarked inputs of the three systems. The benchmark uses 100-bit payloads for TrustMark Q and InvisMark and 32-bit payloads for Watermark Anything. Image-quality and perceptual-distance measurements in Section 6 use the corresponding saved watermarked input as the full-reference image.

The 30-source subset for the joint PI-detection study is selected independently of detector outcomes, BER, perceptual metrics, and previous human judgments. The study includes all three systems and all five removal levels for every selected source.

The 450 image pairs use the exact attacked outputs already saved during the benchmark evaluation, including detected and undetected watermarks. Outputs are not regenerated or selected according to appearance. Ten observers drawn from the first psychovisual study evaluate each pair once.

For this joint study, identity is classified as preserved when the proportion of affirmative responses is at least `0.80`, lost when it is at most `0.30`, and in transition otherwise. These observer-agreement thresholds apply to human responses. They are distinct from the DreamSim-distance thresholds estimated through the logistic calibration and from the SAPIRE parameter `sr`.

## Preprocessing, exported images, and evaluation inputs

The script behavior includes image preprocessing and JPEG export. Consequently, `sr = 0` skips region removal and inpainting but does not guarantee that the exported pixels match the original input file: resizing and JPEG encoding can still alter the image.

A no-attack reference is not automatically equivalent to an exported `sr = 0` image. The experimental record must define how each reference condition is constructed and which preprocessing and encoding operations it undergoes.

**Table S4. Evaluation information to record for each experimental profile.** These values describe the surrounding evaluation workflow and cannot be inferred from the standalone SAPIRE defaults alone.

| Item | Required record |
| --- | --- |
| Evaluated representation | Whether decoding and metrics use the reconstruction output directly or an exported image that is subsequently reloaded. |
| Image encoding | Actual file format and encoder settings for saved inputs and evaluated outputs, including JPEG quality and subsampling when applicable. Record reliance on package defaults together with the package version. |
| Additional preprocessing | Any resizing, cropping, padding, color conversion, normalization, or re-encoding introduced by WIBE, metric implementations, or evaluation scripts. |
| Full-reference image | The source image for the unwatermarked SAPIRE studies; the corresponding saved watermarked input for the benchmark comparisons in Section 6. |
| No-attack reference | The exact processing and encoding path used to produce and evaluate the no-attack condition. |
| Output identity | A stable mapping between each evaluated image, its source, configuration, target and effective `sr`, watermarking system where applicable, and result row. |

To distinguish region-editing effects from resizing and encoding effects, document the actual image-processing paths used by the reference and attacked conditions. The final binary LaMa mask, the pre-refinement union, and the visualization overlay represent different objects and should retain their distinct meanings in exported metadata and analyses.

## Reproducibility records

The complete run configuration is the authoritative record of an experiment. The default values in Table S1 do not establish which overrides or environment-dependent operations were used in a particular run.

For each experimental profile, archive:

1. **Code and configuration:** the script revision, complete `AttackConfig`, command lines, coefficient configuration, target removal levels, and any modifications to the released implementation.
2. **Input manifests:** the COCO release and split, exact image identifiers, subset membership, input dimensions, selection procedure, and any random seeds used for selection. Identify the 30-source subset within the 100-image watermark benchmark.
3. **Model provenance:** the SAM2 repository revision, backbone configuration, checkpoint identifiers or hashes, LaMa wrapper version, and actual LaMa checkpoint.
4. **Execution environment:** Python and package versions, hardware, numerical precision, requested and actual devices for each backend, and any fallback behavior.
5. **Optional operations:** availability of the SAM2 `_C` extension, whether its associated mask post-processing executed, and any relevant runtime warnings.
6. **Output provenance:** the evaluated image representation, any intermediate files, image-encoding settings, and a mapping from outputs to per-image results and human judgments.
7. **Randomness:** seeds and stochastic settings wherever used, including dataset selection, presentation randomization, and downstream resampling analyses.

Passing `min_mask_region_area = 400` to SAM2 does not, by itself, establish that the corresponding optional post-processing was performed. If the extension was unavailable or an operation was skipped, report this together with the package revision and affected experimental runs.

The exported SAPIRE JSON metadata records the transformation information listed in Table S1. It does not automatically provide all environment, evaluation, or dataset-provenance records listed above; archive those separately.

This document covers SAPIRE generation and its connection to the manuscript's experimental profiles. WIBE versions and decision rules, baseline-attack configurations and execution adaptations, metric settings, and statistical analysis procedures must also be supplied with the corresponding evaluation resources.
