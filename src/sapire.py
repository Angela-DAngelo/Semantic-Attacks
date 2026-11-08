"""WIBE adapter for the supplied SAPIRE configuration B (version 1).

Install in ~/sapire-wibe/wibe/user_plugins/sapire.py. Uses the unchanged
~/sapire-wibe/sapire_core.py, SAM2 small and big-LaMa checkpoints.

WIBE interface: SAPIRE(sr=0.30, device="cuda", project_dir=None, seed=0).
Create five attack entries, with distinct report_name values, for
sr = 0.30, 0.45, 0.60, 0.75, 0.90. Run one GPU worker initially.

Input/output: RGB CHW floating tensors in [0, 1]. Input is explicitly
rounded to uint8; output is a CPU float32 tensor. No resizing is done here:
prepare images with longest side <=1024 BEFORE watermark embedding.
An empty selected mask returns the quantized input, as in the source's
8-bit image workflow. This conversion must be considered in the baseline.

The five instances share models and a bounded, last-image cache within a
worker process. Reuse is effective when conditions for the same image are
adjacent; it never reuses segmentation from a different watermarked image.
The cache is in memory only, not a durable checkpoint. Native WIBE output
and experiment-level resume are managed separately by the runner.

No semantic reports, masks or images are written by this plugin. Exceptions
propagate to WIBE. Only a genuinely empty selected mask is a no-op.
The optional --smoke-test checks the adapter with real models on the VM;
its synthetic image is not part of the 100-image experimental dataset.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import threading
from contextlib import contextmanager

from wibench.common.attacks import BaseAttack


SR_VALUES = (0.30, 0.45, 0.60, 0.75, 0.90)
CORE_SHA256 = "4dbe7f5aacaa5f3cacb97039c13d9639b4fb5f0c2c29d923825f96d6454511f1"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
_LOCK = threading.RLock()
_ENGINES = {}


def _load_core(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != CORE_SHA256:
        raise RuntimeError(f"SAPIRE core checksum differs from the validated source: {path}")
    module_name = "_wibe_sapire_core_" + CORE_SHA256[:16]
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load SAPIRE core: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Required by dataclasses.
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return sys.modules[module_name]


@contextmanager
def _inference_context(seed, device):
    """Isolate attack randomness from watermark/payload randomness."""
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cudnn = torch.backends.cudnn
    old_benchmark, old_deterministic = cudnn.benchmark, cudnn.deterministic
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cudnn.benchmark, cudnn.deterministic = False, True
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        cudnn.benchmark, cudnn.deterministic = old_benchmark, old_deterministic


@contextmanager
def _lama_checkpoint(path):
    previous = os.environ.get("LAMA_MODEL")
    os.environ["LAMA_MODEL"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LAMA_MODEL", None)
        else:
            os.environ["LAMA_MODEL"] = previous


class _Engine:
    def __init__(self, project_dir, device, seed):
        self.core = _load_core(project_dir / "sapire_core.py")
        sam_checkpoint = project_dir / "sam2_repo/checkpoints/sam2.1_hiera_small.pt"
        self.lama_checkpoint = project_dir / "models/big-lama.pt"
        for path in (sam_checkpoint, self.lama_checkpoint):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing checkpoint: {path}")
        self.cfg = self.core.AttackConfig()
        self.segmenter = self.core.SAM2AutoMaskGeneratorBackend(
            SAM2_CONFIG, str(sam_checkpoint), device=device
        )
        self.inpainter = self.core.SimpleLaMaBackend(
            binarize_threshold=self.cfg.lama_binarize_threshold, device=device
        )
        self.device, self.seed = device, seed
        self.image_key = None
        self.image = None
        self.regions = []
        self.outputs = {}

    def apply(self, rgb, sr):
        import numpy as np
        from PIL import Image

        key = (rgb.shape, hashlib.sha256(rgb.tobytes()).digest())
        if key != self.image_key:
            # Invalidate first: a failed new image must not poison the cache.
            self.image_key, self.image = None, None
            self.regions, self.outputs = [], {}
            image = Image.fromarray(rgb, mode="RGB")
            with _inference_context(self.seed, self.device):
                raw = self.segmenter.generate(image)
                regions = self.core.build_regions_from_raw_masks(raw, image, self.cfg)
                regions = self.core.filter_regions(regions, self.cfg)
                regions = self.core.score_regions(regions, self.cfg)
            self.image_key, self.image, self.regions = key, image, regions

        info = self.core.make_attack_mask(self.image, self.regions, sr, self.cfg)
        mask_key = (info["empty"], info["mask_key"])
        if mask_key not in self.outputs:
            if info["empty"]:
                result = self.image.copy()
            else:
                with _inference_context(self.seed, self.device), _lama_checkpoint(self.lama_checkpoint):
                    result = self.inpainter.inpaint(self.image, info["mask"])
            if result.size != self.image.size:
                raise ValueError(f"SAPIRE changed image size: {self.image.size} -> {result.size}")
            self.outputs[mask_key] = np.array(result.convert("RGB"), dtype=np.uint8)
        return self.outputs[mask_key].copy()


class SAPIRE(BaseAttack):
    """Native WIBE attack; configuration B and the five agreed sr values."""

    def __init__(self, sr=0.30, device="cuda", project_dir=None, seed=0):
        sr = float(sr)
        matches = [value for value in SR_VALUES if math.isclose(sr, value, rel_tol=0, abs_tol=1e-9)]
        if not matches:
            raise ValueError(f"sr must be one of {SR_VALUES}; received {sr!r}")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.sr = matches[0]
        self.device = str(device)
        self.seed = seed
        self.project_dir = str(Path(project_dir or Path.home() / "sapire-wibe").expanduser().resolve())

    def __call__(self, image):
        import torch

        if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("SAPIRE expects one RGB tensor with shape (3, H, W)")
        if not image.is_floating_point():
            raise TypeError("SAPIRE expects a floating tensor in [0, 1]")
        if min(image.shape[1:]) < 1 or max(image.shape[1:]) > 1024:
            raise ValueError("Prepare image sizes <=1024 before watermark embedding; SAPIRE does not resize")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("SAPIRE input contains NaN or infinity")
        if float(image.min()) < -1e-6 or float(image.max()) > 1.0 + 1e-6:
            raise ValueError("SAPIRE input is outside [0, 1]")

        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only cpu and cuda devices are supported")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
            if device.index >= torch.cuda.device_count():
                raise ValueError(f"Unavailable device: {device}")

        # Explicit, deterministic RGB8 boundary. Never modify WIBE's input.
        rgb = (
            image.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
            .mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        )
        engine_key = (os.getpid(), self.project_dir, str(device), self.seed)
        with _LOCK:
            if engine_key not in _ENGINES:
                _ENGINES[engine_key] = _Engine(Path(self.project_dir), str(device), self.seed)
            result = _ENGINES[engine_key].apply(rgb, self.sr)
        return torch.from_numpy(result).permute(2, 0, 1).contiguous().to(torch.float32).div_(255)


def _smoke_test(project_dir, device):
    """Real-backend check, with no dataset or result files created."""
    import time
    import numpy as np
    import torch

    print("SAPIRE_SMOKE_TEST_START", flush=True)
    # Odd dimensions also exercise LaMa's padding removal.
    yy, xx = np.indices((257, 321))
    rgb = np.stack(((xx * 3 + yy) % 256, (yy * 2) % 256, (xx + yy * 3) % 256), axis=-1).astype(np.uint8)
    rgb[48:155, 65:175] = (225, 45, 30)
    rgb[155:225, 205:285] = (30, 185, 70)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255)
    original = image.clone()
    attacks = [SAPIRE(sr=sr, device=device, project_dir=project_dir) for sr in SR_VALUES]
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    results = []
    for attack in attacks:
        started = time.perf_counter()
        result = attack(image)
        assert result.shape == image.shape and result.dtype == torch.float32
        assert result.device.type == "cpu" and bool(torch.isfinite(result).all())
        assert float(result.min()) >= 0 and float(result.max()) <= 1
        assert torch.equal(image, original), "Attack modified its input"
        results.append(result)
        print(f"sr={attack.sr:.2f} OK | shape={tuple(result.shape)} | seconds={time.perf_counter()-started:.2f}", flush=True)
    assert torch.equal(attacks[0](image), results[0]), "Cached repeat differs"

    # Synthetic segmentation may retain no regions. Independently exercise
    # real LaMa with a non-empty mask to ensure the whole model path works.
    engine = next(value for key, value in _ENGINES.items() if key[0] == os.getpid())
    from PIL import Image
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[75:130, 95:150] = 255
    with _LOCK, _inference_context(0, engine.device), _lama_checkpoint(engine.lama_checkpoint):
        reconstructed = engine.inpainter.inpaint(Image.fromarray(rgb), Image.fromarray(mask))
    assert reconstructed.size == (rgb.shape[1], rgb.shape[0])
    print("LAMA_NONEMPTY_MASK_OK", flush=True)
    if str(device).startswith("cuda"):
        print(f"GPU_PEAK_ALLOCATED_MiB={torch.cuda.max_memory_allocated(device)/1024**2:.1f}", flush=True)
    print("PLUGIN_SAPIRE_OK", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--project-dir", default=str(Path.home() / "sapire-wibe"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.smoke_test:
        parser.error("Use --smoke-test, or import SAPIRE from WIBE")
    _smoke_test(args.project_dir, args.device)
