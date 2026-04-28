"""
Python-side wrapper around the nanobind ``_sd_native`` extension.

The native module exposes raw bindings only. This module adds:
  - IntEnum classes for the C enum types
  - SDImage helpers: PPM/BMP I/O, PIL/numpy round-tripping, ``save()`` dispatcher
  - ``SDContext.generate(...)`` convenience method (the verbose kwargs version)
  - ``text_to_image`` / ``text_to_images`` / ``image_to_image`` helpers
  - ``convert_model`` Python wrapper (validates input, raises FileNotFoundError)
  - ``set_init_image`` / ``set_mask_image`` / ``set_control_image`` /
    ``set_ref_images`` / ``set_pm_id_images`` on SDImageGenParams that keep
    SDImage references alive on the Python side

The split keeps the C++ TU small (~700 lines vs the prior ~3300-line .pyx) while
preserving the ``inferna.sd`` public surface that callers and tests rely on.
"""

from __future__ import annotations

import os
import struct
import logging
from enum import IntEnum
from typing import Callable, List, Optional, Union

from . import _sd_native as _n

_logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Enums (sourced from the native module's ENUMS dict — single source of truth)
# =============================================================================

_E = _n.ENUMS


class RngType(IntEnum):
    STD_DEFAULT = _E["STD_DEFAULT_RNG"]
    CUDA = _E["CUDA_RNG"]
    CPU = _E["CPU_RNG"]


class SampleMethod(IntEnum):
    EULER = _E["EULER_SAMPLE_METHOD"]
    EULER_A = _E["EULER_A_SAMPLE_METHOD"]
    HEUN = _E["HEUN_SAMPLE_METHOD"]
    DPM2 = _E["DPM2_SAMPLE_METHOD"]
    DPMPP2S_A = _E["DPMPP2S_A_SAMPLE_METHOD"]
    DPMPP2M = _E["DPMPP2M_SAMPLE_METHOD"]
    DPMPP2Mv2 = _E["DPMPP2Mv2_SAMPLE_METHOD"]
    IPNDM = _E["IPNDM_SAMPLE_METHOD"]
    IPNDM_V = _E["IPNDM_V_SAMPLE_METHOD"]
    LCM = _E["LCM_SAMPLE_METHOD"]
    DDIM_TRAILING = _E["DDIM_TRAILING_SAMPLE_METHOD"]
    TCD = _E["TCD_SAMPLE_METHOD"]
    RES_MULTISTEP = _E["RES_MULTISTEP_SAMPLE_METHOD"]
    RES_2S = _E["RES_2S_SAMPLE_METHOD"]
    ER_SDE = _E["ER_SDE_SAMPLE_METHOD"]
    COUNT = _E["SAMPLE_METHOD_COUNT"]


class Scheduler(IntEnum):
    DISCRETE = _E["DISCRETE_SCHEDULER"]
    KARRAS = _E["KARRAS_SCHEDULER"]
    EXPONENTIAL = _E["EXPONENTIAL_SCHEDULER"]
    AYS = _E["AYS_SCHEDULER"]
    GITS = _E["GITS_SCHEDULER"]
    SGM_UNIFORM = _E["SGM_UNIFORM_SCHEDULER"]
    SIMPLE = _E["SIMPLE_SCHEDULER"]
    SMOOTHSTEP = _E["SMOOTHSTEP_SCHEDULER"]
    KL_OPTIMAL = _E["KL_OPTIMAL_SCHEDULER"]
    LCM = _E["LCM_SCHEDULER"]
    BONG_TANGENT = _E["BONG_TANGENT_SCHEDULER"]
    COUNT = _E["SCHEDULER_COUNT"]


class Prediction(IntEnum):
    EPS = _E["EPS_PRED"]
    V = _E["V_PRED"]
    EDM_V = _E["EDM_V_PRED"]
    FLOW = _E["FLOW_PRED"]
    FLUX_FLOW = _E["FLUX_FLOW_PRED"]
    FLUX2_FLOW = _E["FLUX2_FLOW_PRED"]
    COUNT = _E["PREDICTION_COUNT"]


class SDType(IntEnum):
    F32 = _E["SD_TYPE_F32"]
    F16 = _E["SD_TYPE_F16"]
    Q4_0 = _E["SD_TYPE_Q4_0"]
    Q4_1 = _E["SD_TYPE_Q4_1"]
    Q5_0 = _E["SD_TYPE_Q5_0"]
    Q5_1 = _E["SD_TYPE_Q5_1"]
    Q8_0 = _E["SD_TYPE_Q8_0"]
    Q8_1 = _E["SD_TYPE_Q8_1"]
    Q2_K = _E["SD_TYPE_Q2_K"]
    Q3_K = _E["SD_TYPE_Q3_K"]
    Q4_K = _E["SD_TYPE_Q4_K"]
    Q5_K = _E["SD_TYPE_Q5_K"]
    Q6_K = _E["SD_TYPE_Q6_K"]
    Q8_K = _E["SD_TYPE_Q8_K"]
    BF16 = _E["SD_TYPE_BF16"]
    COUNT = _E["SD_TYPE_COUNT"]


class LogLevel(IntEnum):
    DEBUG = _E["SD_LOG_DEBUG"]
    INFO = _E["SD_LOG_INFO"]
    WARN = _E["SD_LOG_WARN"]
    ERROR = _E["SD_LOG_ERROR"]


class PreviewMode(IntEnum):
    NONE = _E["PREVIEW_NONE"]
    PROJ = _E["PREVIEW_PROJ"]
    TAE = _E["PREVIEW_TAE"]
    VAE = _E["PREVIEW_VAE"]


class LoraApplyMode(IntEnum):
    AUTO = _E["LORA_APPLY_AUTO"]
    IMMEDIATELY = _E["LORA_APPLY_IMMEDIATELY"]
    AT_RUNTIME = _E["LORA_APPLY_AT_RUNTIME"]


class HiresUpscaler(IntEnum):
    NONE = _E["SD_HIRES_UPSCALER_NONE"]
    LATENT = _E["SD_HIRES_UPSCALER_LATENT"]
    LATENT_NEAREST = _E["SD_HIRES_UPSCALER_LATENT_NEAREST"]
    LATENT_NEAREST_EXACT = _E["SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT"]
    LATENT_ANTIALIASED = _E["SD_HIRES_UPSCALER_LATENT_ANTIALIASED"]
    LATENT_BICUBIC = _E["SD_HIRES_UPSCALER_LATENT_BICUBIC"]
    LATENT_BICUBIC_ANTIALIASED = _E["SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED"]
    LANCZOS = _E["SD_HIRES_UPSCALER_LANCZOS"]
    NEAREST = _E["SD_HIRES_UPSCALER_NEAREST"]
    MODEL = _E["SD_HIRES_UPSCALER_MODEL"]


# =============================================================================
# SDImage — composition wrapper around the native handle.
# We don't subclass _n.SDImage because nanobind enforces matching deallocators
# (Python subclasses get a different tp_dealloc, blocking __class__ reassign).
# =============================================================================

class SDImage:
    """Pillow/numpy-friendly wrapper around the stb-backed native SDImage."""

    __slots__ = ("_native",)

    def __init__(self, _native=None):
        self._native = _native if _native is not None else _n.SDImage()

    # ---- core attribute proxies ----------------------------------------------

    @property
    def width(self) -> int: return self._native.width
    @property
    def height(self) -> int: return self._native.height
    @property
    def channels(self) -> int: return self._native.channels
    @property
    def is_valid(self) -> bool: return self._native.is_valid
    @property
    def shape(self) -> tuple: return (self.height, self.width, self.channels)
    @property
    def size(self) -> int: return self.width * self.height * self.channels

    # ---- numpy / PIL ----------------------------------------------------------

    def to_numpy(self):
        if not HAS_NUMPY:
            raise ImportError("numpy is required for to_numpy()")
        return self._native.to_numpy()

    def to_pil(self):
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for to_pil()")
        arr = self.to_numpy()
        c = self.channels
        if c == 3:
            return PILImage.fromarray(arr, mode="RGB")
        if c == 4:
            return PILImage.fromarray(arr, mode="RGBA")
        if c == 1:
            return PILImage.fromarray(arr[:, :, 0], mode="L")
        return PILImage.fromarray(arr)

    @classmethod
    def from_numpy(cls, arr) -> "SDImage":
        if not HAS_NUMPY:
            raise ImportError("numpy is required for from_numpy()")
        return cls(_n.SDImage.from_numpy(arr))

    @classmethod
    def from_pil(cls, pil_image) -> "SDImage":
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for from_pil()")
        if not HAS_NUMPY:
            raise ImportError("numpy is required for from_pil()")
        if pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGB")
        arr = np.array(pil_image)
        return cls.from_numpy(arr)

    # ---- stb-backed save (delegated to native) -------------------------------

    def save_png(self, path: str) -> None:
        self._native.save_png(path)

    def save_jpg(self, path: str, quality: int = 90) -> None:
        self._native.save_jpg(path, quality)

    # ---- PPM / BMP I/O (pure-python; no stb dependency) ----------------------

    def save_ppm(self, path: str) -> None:
        if not self.is_valid:
            raise ValueError("Image has no valid data")
        w, h, c = self.width, self.height, self.channels
        arr = self.to_numpy()
        with open(path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            if c == 3:
                f.write(arr.tobytes())
            elif c == 4:
                f.write(arr[:, :, :3].tobytes())
            elif c == 1:
                rgb = np.repeat(arr, 3, axis=2)
                f.write(rgb.tobytes())
            else:
                raise ValueError(f"Unsupported channel count: {c}")

    def save_bmp(self, path: str) -> None:
        if not self.is_valid:
            raise ValueError("Image has no valid data")
        w, h, c = self.width, self.height, self.channels
        arr = self.to_numpy()
        # 24-bit BMP writer; rows padded to 4-byte alignment, BGR pixel order,
        # bottom-up scanline order.
        row_bytes = w * 3
        padding = (4 - (row_bytes % 4)) % 4
        padded_row = row_bytes + padding
        pixel_data_size = padded_row * h
        file_size = 54 + pixel_data_size
        with open(path, "wb") as f:
            f.write(b"BM")
            f.write(struct.pack("<I", file_size))
            f.write(struct.pack("<HH", 0, 0))
            f.write(struct.pack("<I", 54))
            f.write(struct.pack("<I", 40))
            f.write(struct.pack("<i", w))
            f.write(struct.pack("<i", h))
            f.write(struct.pack("<HH", 1, 24))
            f.write(struct.pack("<I", 0))
            f.write(struct.pack("<I", pixel_data_size))
            f.write(struct.pack("<i", 2835))
            f.write(struct.pack("<i", 2835))
            f.write(struct.pack("<I", 0))
            f.write(struct.pack("<I", 0))
            pad_bytes = b"\x00" * padding
            for y in range(h - 1, -1, -1):
                row = arr[y]
                if c >= 3:
                    bgr = row[:, [2, 1, 0]].tobytes()
                else:
                    bgr = np.repeat(row, 3, axis=1).tobytes()
                f.write(bgr)
                if padding:
                    f.write(pad_bytes)

    @classmethod
    def load_ppm(cls, path: str) -> "SDImage":
        with open(path, "rb") as f:
            magic = f.readline().strip()
            if magic != b"P6":
                raise ValueError(
                    f"Unsupported PPM format: {magic}. Only P6 (binary RGB) supported."
                )
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            parts = line.strip().split()
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
            else:
                w = int(parts[0])
                h = int(f.readline().strip())
            max_val = int(f.readline().strip())
            if max_val != 255:
                raise ValueError(f"Unsupported max value: {max_val}. Only 255 supported.")
            data = f.read(w * h * 3)
        if not HAS_NUMPY:
            raise ImportError("numpy is required for load_ppm()")
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
        return cls.from_numpy(arr)

    @classmethod
    def load_bmp(cls, path: str) -> "SDImage":
        with open(path, "rb") as f:
            sig = f.read(2)
            if sig != b"BM":
                raise ValueError(f"Not a BMP file: {sig!r}")
            f.read(4)  # file size
            f.read(4)  # reserved
            pixel_offset = struct.unpack("<I", f.read(4))[0]
            f.read(4)  # header size
            w = struct.unpack("<i", f.read(4))[0]
            h_signed = struct.unpack("<i", f.read(4))[0]
            top_down = h_signed < 0
            h = abs(h_signed)
            struct.unpack("<H", f.read(2))[0]  # planes
            bpp = struct.unpack("<H", f.read(2))[0]
            compression = struct.unpack("<I", f.read(4))[0]
            if bpp != 24:
                raise ValueError(f"Unsupported BMP bit depth: {bpp}. Only 24-bit supported.")
            if compression != 0:
                raise ValueError("Compressed BMP not supported.")
            f.seek(pixel_offset)
            row_bytes = w * 3
            padding = (4 - (row_bytes % 4)) % 4
            padded_row = row_bytes + padding
            if not HAS_NUMPY:
                raise ImportError("numpy is required for load_bmp()")
            out = np.empty((h, w, 3), dtype=np.uint8)
            for y in range(h):
                target_y = y if top_down else (h - 1 - y)
                row = f.read(padded_row)[:row_bytes]
                row_arr = np.frombuffer(row, dtype=np.uint8).reshape((w, 3))
                # Stored as BGR; flip to RGB.
                out[target_y] = row_arr[:, [2, 1, 0]]
        return cls.from_numpy(out)

    @classmethod
    def load(cls, path: str, channels: int = 0) -> "SDImage":
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        if ext == "ppm":
            return cls.load_ppm(path)
        return cls(_n.SDImage.load_stb(path, channels))

    def save(self, path: str, quality: int = 90) -> None:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        if ext == "png":
            self.save_png(path); return
        if ext in ("jpg", "jpeg"):
            self.save_jpg(path, quality); return
        if ext == "bmp":
            self.save_bmp(path); return
        if ext == "ppm":
            self.save_ppm(path); return
        if HAS_PIL:
            self.to_pil().save(path)
            return
        # Unknown extension and no PIL — fall back to PNG.
        if ext in ("gif", "tiff", "webp"):
            png_path = path.rsplit(".", 1)[0] + ".png"
            self.save_png(png_path)
            raise ImportError(
                f"PIL/Pillow required for {ext.upper()} format. "
                f"Image saved as PNG instead: {png_path}"
            )
        self.save_png(path)


# =============================================================================
# SDContextParams — re-export with kwargs constructor sugar
# =============================================================================

class SDContextParams(_n.SDContextParams):
    """Constructor kwargs are a convenience over the native default-init + setters."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        clip_l_path: Optional[str] = None,
        clip_g_path: Optional[str] = None,
        t5xxl_path: Optional[str] = None,
        diffusion_model_path: Optional[str] = None,
        n_threads: int = -1,
        wtype: SDType = SDType.COUNT,
        vae_decode_only: bool = True,
    ):
        super().__init__()
        if model_path: self.model_path = model_path
        if vae_path: self.vae_path = vae_path
        if clip_l_path: self.clip_l_path = clip_l_path
        if clip_g_path: self.clip_g_path = clip_g_path
        if t5xxl_path: self.t5xxl_path = t5xxl_path
        if diffusion_model_path: self.diffusion_model_path = diffusion_model_path
        if n_threads > 0:
            self.n_threads = n_threads
        self.wtype = int(wtype)
        self.vae_decode_only = vae_decode_only


# =============================================================================
# SDSampleParams — kwargs sugar + ctor wiring (eta default = +inf)
# =============================================================================

class SDSampleParams(_n.SDSampleParams):
    def __init__(
        self,
        sample_method: SampleMethod = SampleMethod.COUNT,
        scheduler: Scheduler = Scheduler.COUNT,
        sample_steps: int = 20,
        cfg_scale: float = 7.0,
        eta: float = float("inf"),
    ):
        super().__init__()
        self.sample_method = int(sample_method)
        self.scheduler = int(scheduler)
        self.sample_steps = sample_steps
        self.cfg_scale = cfg_scale
        self.eta = eta


# =============================================================================
# SDImageGenParams — kwargs sugar + Python-side image-set helpers that retain
# refs to SDImage instances on the wrapper (so their data buffers persist).
# =============================================================================

class SDImageGenParams(_n.SDImageGenParams):
    def __init__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        batch_count: int = 1,
        sample_steps: int = 20,
        cfg_scale: float = 7.0,
        sample_method: SampleMethod = SampleMethod.COUNT,
        scheduler: Scheduler = Scheduler.COUNT,
        strength: float = 0.75,
        clip_skip: int = -1,
    ):
        super().__init__()
        self._init_image_ref: Optional[SDImage] = None
        self._mask_image_ref: Optional[SDImage] = None
        self._control_image_ref: Optional[SDImage] = None
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.seed = seed
        self.batch_count = batch_count
        self.strength = strength
        self.clip_skip = clip_skip
        sp = self.sample_params  # reference into the embedded sample struct
        sp.sample_method = int(sample_method)
        sp.scheduler = int(scheduler)
        sp.sample_steps = sample_steps
        sp.cfg_scale = cfg_scale

    # ---- image setters --------------------------------------------------------

    def set_init_image(self, image: SDImage) -> None:
        self._init_image_ref = image
        super().set_init_image(image._native, image)

    def set_mask_image(self, image: SDImage) -> None:
        self._mask_image_ref = image
        super().set_mask_image(image._native, image)

    def set_control_image(self, image: SDImage, strength: float = 1.0) -> None:
        self._control_image_ref = image
        super().set_control_image(image._native, image)
        self.control_strength = strength

    def set_ref_images(self, images: list) -> None:
        self._ref_images_pyref = list(images)
        super().set_ref_images([img._native for img in images])

    def set_pm_id_images(self, images: list) -> None:
        self._pm_id_images_pyref = list(images)
        super().set_pm_id_images([img._native for img in images])

    # ---- VAE tiling tuple shims (kept for API compatibility) ------------------

    @property
    def vae_tile_size(self) -> tuple:
        return (self.vae_tile_size_x, self.vae_tile_size_y)

    @vae_tile_size.setter
    def vae_tile_size(self, value: tuple) -> None:
        self.vae_tile_size_x, self.vae_tile_size_y = value

    @property
    def vae_tile_rel_size(self) -> tuple:
        return (self.vae_tile_rel_size_x, self.vae_tile_rel_size_y)

    @vae_tile_rel_size.setter
    def vae_tile_rel_size(self, value: tuple) -> None:
        self.vae_tile_rel_size_x, self.vae_tile_rel_size_y = value

    @property
    def hires_target_size(self) -> tuple:
        return (self.hires_target_width, self.hires_target_height)

    @hires_target_size.setter
    def hires_target_size(self, value: tuple) -> None:
        self.hires_target_width, self.hires_target_height = value

    # ---- cache_range / easycache_* aliases ------------------------------------

    @property
    def cache_range(self) -> tuple:
        return (self.cache_start_percent, self.cache_end_percent)

    @cache_range.setter
    def cache_range(self, value: tuple) -> None:
        self.cache_start_percent, self.cache_end_percent = value

    @property
    def easycache_enabled(self) -> bool:
        return self.cache_mode == _E["SD_CACHE_EASYCACHE"]

    @easycache_enabled.setter
    def easycache_enabled(self, value: bool) -> None:
        self.cache_mode = _E["SD_CACHE_EASYCACHE"] if value else _E["SD_CACHE_DISABLED"]

    @property
    def easycache_threshold(self) -> float:
        return self.cache_threshold

    @easycache_threshold.setter
    def easycache_threshold(self, value: float) -> None:
        self.cache_threshold = value

    @property
    def easycache_range(self) -> tuple:
        return self.cache_range

    @easycache_range.setter
    def easycache_range(self, value: tuple) -> None:
        self.cache_range = value

    # ---- hires-fix one-shot setter -------------------------------------------

    def set_hires_fix(
        self,
        enabled: bool = True,
        upscaler: Optional[int] = None,
        scale: float = 2.0,
        model_path: Optional[str] = None,
        target_width: int = 0,
        target_height: int = 0,
        steps: int = 0,
        denoising_strength: float = 0.7,
        tile_size: int = 128,
    ) -> None:
        self.hires_enabled = enabled
        self.hires_upscaler = int(upscaler) if upscaler is not None else _E["SD_HIRES_UPSCALER_LATENT"]
        self.hires_scale = scale
        self.hires_model_path = model_path
        self.hires_target_size = (target_width, target_height)
        self.hires_steps = steps
        self.hires_denoising_strength = denoising_strength
        self.hires_tile_size = tile_size


# =============================================================================
# SDContext — adds the kwargs-style generate() that builds an SDImageGenParams
# =============================================================================

class SDContext(_n.SDContext):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Deterministic teardown — release the native context as soon as
        # the with-block exits. Without this, the ctx (and its GPU buffers)
        # would linger until GC ran, which on macOS Metal accumulates
        # working-set pressure across consecutive contexts.
        self.close()
        return None

    def get_default_sample_method(self) -> SampleMethod:
        return SampleMethod(super().get_default_sample_method())

    def get_default_scheduler(self, sample_method: Optional[SampleMethod] = None) -> Scheduler:
        sm = int(sample_method) if sample_method is not None else None
        return Scheduler(super().get_default_scheduler(sm))

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        batch_count: int = 1,
        sample_steps: int = 20,
        cfg_scale: float = 7.0,
        sample_method: Optional[SampleMethod] = None,
        scheduler: Optional[Scheduler] = None,
        init_image: Optional[SDImage] = None,
        mask_image: Optional[SDImage] = None,
        control_image: Optional[SDImage] = None,
        control_strength: float = 1.0,
        strength: float = 0.75,
        clip_skip: int = -1,
        eta: float = float("inf"),
        slg_scale: float = 0.0,
        flow_shift: float = float("inf"),
        vae_tiling: bool = False,
        hires_fix: bool = False,
        hires_scale: float = 2.0,
    ) -> List[SDImage]:
        if sample_method is None:
            sample_method = SampleMethod.COUNT
        if scheduler is None:
            scheduler = Scheduler.COUNT
        params = SDImageGenParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            batch_count=batch_count,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            strength=strength,
            clip_skip=clip_skip,
        )
        sp = params.sample_params
        sp.eta = eta
        sp.slg_scale = slg_scale
        sp.flow_shift = flow_shift
        params.vae_tiling_enabled = vae_tiling
        if hires_fix:
            params.set_hires_fix(enabled=True, scale=hires_scale)
        if init_image is not None:
            params.set_init_image(init_image)
        if mask_image is not None:
            params.set_mask_image(mask_image)
        if control_image is not None:
            params.set_control_image(control_image, control_strength)
        return self.generate_with_params(params)

    def generate_with_params(self, params: SDImageGenParams) -> List[SDImage]:
        out = super().generate_with_params(params)
        return [SDImage(native) for native in out]

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        video_frames: int = 16,
        sample_steps: int = 20,
        cfg_scale: float = 7.0,
        sample_method: Optional[SampleMethod] = None,
        scheduler: Optional[Scheduler] = None,
        init_image: Optional[SDImage] = None,
        end_image: Optional[SDImage] = None,
        strength: float = 0.75,
        clip_skip: int = -1,
        eta: float = float("inf"),
        moe_boundary: float = 0.875,
        vace_strength: float = 1.0,
    ) -> List[SDImage]:
        sm = int(sample_method) if sample_method is not None else int(SampleMethod.COUNT)
        sc = int(scheduler) if scheduler is not None else int(Scheduler.COUNT)
        out = self.generate_video_raw(
            prompt, negative_prompt, width, height, video_frames, sm, sc,
            sample_steps, cfg_scale, seed, clip_skip, strength,
            eta, moe_boundary, vace_strength,
            init_image._native if init_image is not None else None,
            end_image._native if end_image is not None else None,
        )
        return [SDImage(native) for native in out]


# =============================================================================
# Re-exports / convenience wrappers
# =============================================================================

class Upscaler:
    """Upscaler wrapper that hands back SDImage instances (not raw native)."""

    def __init__(
        self, model_path: str,
        n_threads: int = -1,
        offload_to_cpu: bool = False,
        direct: bool = False,
        tile_size: int = 0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._native = _n.Upscaler(model_path, offload_to_cpu, direct, n_threads, tile_size)

    @property
    def is_valid(self) -> bool: return self._native.is_valid

    @property
    def upscale_factor(self) -> int: return self._native.upscale_factor

    def upscale(self, image: SDImage, factor: int = 0) -> SDImage:
        return SDImage(self._native.upscale(image._native, factor))

    def __enter__(self): return self

    def __exit__(self, exc_type, exc_val, exc_tb): return None


def get_num_cores() -> int:
    return _n.get_num_cores()


def get_system_info() -> str:
    return _n.get_system_info()


def type_name(t: SDType) -> str:
    return _n.type_name(int(t))


def sample_method_name(m: SampleMethod) -> str:
    return _n.sample_method_name(int(m))


def scheduler_name(s: Scheduler) -> str:
    return _n.scheduler_name(int(s))


def ggml_backend_load_all() -> None:
    _n.ggml_backend_load_all()


def canny_preprocess(
    image: SDImage,
    high_threshold: float = 0.8,
    low_threshold: float = 0.1,
    weak: float = 0.5,
    strong: float = 1.0,
    inverse: bool = False,
) -> bool:
    return _n.preprocess_canny(image._native, high_threshold, low_threshold,
                                weak, strong, inverse)


def convert_model(
    input_path: str,
    output_path: str,
    output_type: SDType = SDType.F16,
    vae_path: Optional[str] = None,
    tensor_type_rules: Optional[str] = None,
    convert_name: bool = False,
) -> bool:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")
    success = _n.convert_native(
        input_path, output_path, int(output_type),
        vae_path, tensor_type_rules, convert_name,
    )
    if not success:
        raise RuntimeError(f"Model conversion failed: {input_path} -> {output_path}")
    return True


# ---- callbacks --------------------------------------------------------------

def set_log_callback(callback: Optional[Callable[[LogLevel, str], None]]) -> None:
    if callback is None:
        _n.set_log_callback(None)
        return
    def _wrap(level_int: int, text: str) -> None:
        try:
            callback(LogLevel(level_int), text)
        except Exception as e:
            _logger.warning("Exception in sd log callback: %s", e)
    _n.set_log_callback(_wrap)


def set_progress_callback(callback: Optional[Callable[[int, int, float], None]]) -> None:
    if callback is None:
        _n.set_progress_callback(None)
        return
    def _wrap(step: int, steps: int, time: float) -> None:
        try:
            callback(step, steps, time)
        except Exception as e:
            _logger.warning("Exception in sd progress callback: %s", e)
    _n.set_progress_callback(_wrap)


def set_preview_callback(
    callback: Optional[Callable[[int, List[SDImage], bool], None]],
    mode: PreviewMode = PreviewMode.NONE,
    interval: int = 1,
    denoised: bool = True,
    noisy: bool = False,
) -> None:
    if callback is None:
        _n.set_preview_callback(None, int(PreviewMode.NONE), 1, False, False)
        return
    def _wrap(step: int, frames: list, is_noisy: bool) -> None:
        try:
            wrapped = [SDImage(native) for native in frames]
            callback(step, wrapped, is_noisy)
        except Exception as e:
            _logger.warning("Exception in sd preview callback: %s", e)
    _n.set_preview_callback(_wrap, int(mode), interval, denoised, noisy)


# =============================================================================
# Convenience top-level functions
# =============================================================================

def text_to_images(
    model_path: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    batch_count: int = 1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    sample_method: SampleMethod = SampleMethod.COUNT,
    scheduler: Scheduler = Scheduler.COUNT,
    n_threads: int = -1,
    vae_path: Optional[str] = None,
    taesd_path: Optional[str] = None,
    clip_l_path: Optional[str] = None,
    clip_g_path: Optional[str] = None,
    t5xxl_path: Optional[str] = None,
    control_net_path: Optional[str] = None,
    clip_skip: int = -1,
    eta: float = float("inf"),
    slg_scale: float = 0.0,
    vae_tiling: bool = False,
    hires_fix: bool = False,
    hires_scale: float = 2.0,
    offload_to_cpu: bool = False,
    keep_clip_on_cpu: bool = False,
    keep_vae_on_cpu: bool = False,
    diffusion_flash_attn: bool = False,
) -> List[SDImage]:
    params = SDContextParams(
        model_path=model_path,
        vae_path=vae_path,
        clip_l_path=clip_l_path,
        clip_g_path=clip_g_path,
        t5xxl_path=t5xxl_path,
        n_threads=n_threads,
    )
    if taesd_path:
        params.taesd_path = taesd_path
    if control_net_path:
        params.control_net_path = control_net_path
    params.offload_params_to_cpu = offload_to_cpu
    params.keep_clip_on_cpu = keep_clip_on_cpu
    params.keep_vae_on_cpu = keep_vae_on_cpu
    params.diffusion_flash_attn = diffusion_flash_attn

    with SDContext(params) as ctx:
        return ctx.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            batch_count=batch_count,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            eta=eta,
            slg_scale=slg_scale,
            vae_tiling=vae_tiling,
            hires_fix=hires_fix,
            hires_scale=hires_scale,
            clip_skip=clip_skip,
        )


def text_to_image(
    model_path: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    sample_method: SampleMethod = SampleMethod.COUNT,
    scheduler: Scheduler = Scheduler.COUNT,
    n_threads: int = -1,
    vae_path: Optional[str] = None,
    taesd_path: Optional[str] = None,
    clip_l_path: Optional[str] = None,
    clip_g_path: Optional[str] = None,
    t5xxl_path: Optional[str] = None,
    control_net_path: Optional[str] = None,
    clip_skip: int = -1,
    eta: float = float("inf"),
    slg_scale: float = 0.0,
    vae_tiling: bool = False,
    hires_fix: bool = False,
    hires_scale: float = 2.0,
    offload_to_cpu: bool = False,
    keep_clip_on_cpu: bool = False,
    keep_vae_on_cpu: bool = False,
    diffusion_flash_attn: bool = False,
) -> SDImage:
    return text_to_images(
        model_path=model_path, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, seed=seed, batch_count=1,
        sample_steps=sample_steps, cfg_scale=cfg_scale,
        sample_method=sample_method, scheduler=scheduler,
        n_threads=n_threads, vae_path=vae_path, taesd_path=taesd_path,
        clip_l_path=clip_l_path, clip_g_path=clip_g_path, t5xxl_path=t5xxl_path,
        control_net_path=control_net_path, clip_skip=clip_skip,
        eta=eta, slg_scale=slg_scale, vae_tiling=vae_tiling,
        hires_fix=hires_fix, hires_scale=hires_scale,
        offload_to_cpu=offload_to_cpu, keep_clip_on_cpu=keep_clip_on_cpu,
        keep_vae_on_cpu=keep_vae_on_cpu, diffusion_flash_attn=diffusion_flash_attn,
    )[0]


def image_to_image(
    model_path: str,
    init_image: Union[SDImage, str],
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.75,
    seed: int = -1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    sample_method: SampleMethod = SampleMethod.COUNT,
    scheduler: Scheduler = Scheduler.COUNT,
    n_threads: int = -1,
    vae_path: Optional[str] = None,
    clip_skip: int = -1,
) -> List[SDImage]:
    if isinstance(init_image, str):
        init_image = SDImage.load(init_image)
    params = SDContextParams(
        model_path=model_path,
        vae_path=vae_path,
        n_threads=n_threads,
        vae_decode_only=False,
    )
    with SDContext(params) as ctx:
        return ctx.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=init_image.width,
            height=init_image.height,
            seed=seed,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            init_image=init_image,
            strength=strength,
            clip_skip=clip_skip,
        )
