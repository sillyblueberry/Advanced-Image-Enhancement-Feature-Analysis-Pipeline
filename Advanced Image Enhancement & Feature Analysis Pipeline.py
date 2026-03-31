"""
Advanced Image Enhancement & Feature Analysis Pipeline
CSE3010 - Computer Vision | Final Project
Author  : [Your Name]
Module  : Classical Computer Vision (Modules 1-4)
Constraint: No deep learning — only classical CV techniques.

Pipeline Stages:
    1. Preprocessing   → Gaussian Blur + Median Filter (noise reduction)
    2. Enhancement     → CLAHE in LAB colour space (local contrast)
    3. Restoration     → Unsharp Masking / Laplacian Sharpening
    4. Feature Analysis→ Harris Corner Detection + ORB Feature Extraction
    5. Edge Analysis   → Canny Edge Detection (bonus stage)

CLI Modes: demo | process
"""

import cv2
import numpy as np
import argparse
import sys
import time
import warnings
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Guarantee the image is in 3-channel BGR format.
    Handles: grayscale (H×W), BGR (H×W×3), BGRA (H×W×4).
    """
    if image is None:
        raise ValueError("ensure_bgr received None — check image loading.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def letterbox_resize(image: np.ndarray,
                     target_w: int,
                     target_h: int,
                     pad_color: tuple = (20, 20, 20)) -> np.ndarray:
    """
    Resize image to (target_w × target_h) while preserving aspect ratio.
    Padding fills the empty space — avoids the distortion of a plain stretch.
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 – PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian Blur for frequency-domain noise suppression.

    Convolves the image with a Gaussian kernel:
        G(x,y) = (1 / 2πσ²) · exp(−(x² + y²) / 2σ²)

    Effective against high-frequency Gaussian noise while preserving
    low-frequency (structural) content.

    Args:
        image : Input BGR image.
        ksize : Kernel size (odd integer). Default 5×5.
        sigma : Std-dev; 0 = auto-derived from ksize per OpenCV formula.

    Returns:
        Gaussian-blurred BGR image (same dtype/shape).
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Median Filtering for impulse (salt-and-pepper) noise removal.

    Each pixel is replaced by the median of its k×k neighbourhood.
    Non-linear operator — preserves edges far better than mean filters
    while aggressively eliminating isolated spike noise.

    Args:
        image : BGR image (post Gaussian blur).
        ksize : Neighbourhood size (odd, ≥ 3). Default 3×3.

    Returns:
        Median-filtered BGR image.
    """
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)
    return cv2.medianBlur(image, ksize)


def preprocess(image: np.ndarray,
               gauss_ksize: int = 5,
               median_ksize: int = 3) -> np.ndarray:
    """
    Stage 1 — Full preprocessing: Gaussian → Median chain.

    Gaussian blur first attenuates wideband random noise; the subsequent
    median pass removes residual impulse artefacts without further blurring
    structural edges.

    Args:
        image        : Raw input image (any valid OpenCV format).
        gauss_ksize  : Gaussian kernel size.
        median_ksize : Median kernel size.

    Returns:
        Denoised BGR image ready for enhancement.
    """
    image = ensure_bgr(image)
    blurred  = gaussian_blur(image, ksize=gauss_ksize)
    filtered = median_filter(blurred, ksize=median_ksize)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 – ENHANCEMENT  (CLAHE)
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Stage 2 — CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Operates on the L-channel in CIE-LAB colour space to boost local contrast
    without distorting hue or saturation. The clip limit caps histogram bins
    before redistribution, preventing noise over-amplification in near-uniform
    regions — the principal failure mode of global HE.

    Algorithm:
        1. BGR → LAB (perceptual colour space).
        2. CLAHE on L-channel with clip_limit and tile_grid parameters.
        3. Merge enhanced L with original A, B; convert back to BGR.

    Args:
        image      : Preprocessed BGR image.
        clip_limit : Amplification ceiling (recommended 1.5–4.0).
        tile_grid  : (cols, rows) tile count for contextual histograms.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 – RESTORATION  (Sharpening)
# ─────────────────────────────────────────────────────────────────────────────

def unsharp_mask(image: np.ndarray,
                 blur_ksize: int = 5,
                 strength: float = 1.5,
                 threshold: int = 0) -> np.ndarray:
    """
    Stage 3a — Unsharp Masking (high-frequency detail restoration).

    Formula:  Sharpened = Original·(1 + α) − Blurred·α

    Isolates high-frequency content (edges, texture) by subtracting a
    low-pass copy, then adds it back with amplification factor α.
    Avoids the ringing artefacts of pure Laplacian sharpening.

    Implementation note: arithmetic is performed in float32 to prevent
    silent uint8 overflow/wrap-around before final clipping.

    Args:
        image      : CLAHE-enhanced BGR image.
        blur_ksize : Gaussian kernel size for the low-pass mask.
        strength   : Edge amplification factor α (default 1.5).
        threshold  : Noise gate — minimum absolute difference required
                     before sharpening is applied (0 = off).

    Returns:
        Sharpened BGR image (uint8).
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    img_f   = image.astype(np.float32)
    blur_f  = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0).astype(np.float32)
    sharp   = img_f * (1.0 + strength) - blur_f * strength
    if threshold > 0:
        gate = np.abs(img_f - blur_f) < threshold
        sharp[gate] = img_f[gate]
    return np.clip(sharp, 0, 255).astype(np.uint8)


def laplacian_sharpen(image: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Stage 3b — Laplacian-based detail sharpening.

    Computes the discrete Laplacian ∇²f (second-derivative approximation),
    which responds strongly at rapid intensity transitions. The result is
    blended back into the original to enhance edge definition.

    Args:
        image : BGR input.
        alpha : Laplacian blend weight (default 0.7).

    Returns:
        Sharpened BGR image (uint8).
    """
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap      = cv2.Laplacian(gray, cv2.CV_64F)
    lap_8u   = cv2.convertScaleAbs(lap)
    lap_bgr  = cv2.cvtColor(lap_8u, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(image, 1.0, lap_bgr, alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def restore(image: np.ndarray, method: str = "unsharp", **kwargs) -> np.ndarray:
    """
    Restoration dispatcher — selects sharpening backend.

    Args:
        image  : CLAHE-enhanced BGR image.
        method : 'unsharp' (default) | 'laplacian'
        kwargs : Forwarded verbatim to the selected function.

    Returns:
        Restored/sharpened BGR image.
    """
    if method == "laplacian":
        return laplacian_sharpen(image, **kwargs)
    return unsharp_mask(image, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 – FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def harris_corners(image: np.ndarray,
                   block_size: int = 2,
                   ksize: int = 3,
                   k: float = 0.04,
                   threshold_ratio: float = 0.01) -> tuple:
    """
    Stage 4a — Harris Corner Detection.

    Computes the Harris response:
        R = det(M) − k · trace²(M)
    where M is the 2×2 structure tensor built from local image gradients.
    R > 0 → corner  |  R < 0 → edge  |  |R| ≈ 0 → flat region.

    FIX (vs naive np.sum): Corner count uses cv2.connectedComponents to
    count distinct corner *regions*, not raw pixels. Dilation inflates a
    single corner into many pixels; component labelling collapses them back.

    Args:
        image          : BGR restored image.
        block_size     : Neighbourhood integration window.
        ksize          : Sobel derivative kernel size.
        k              : Harris sensitivity (Harris & Stephens, 1988).
        threshold_ratio: Detection threshold as fraction of max response.

    Returns:
        (corner_image, num_corners)
    """
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = cv2.cornerHarris(gray, block_size, ksize, k)
    dilated  = cv2.dilate(response, None)

    mask        = (dilated > threshold_ratio * dilated.max()).astype(np.uint8)
    corner_img  = image.copy()
    corner_img[mask == 1] = [0, 0, 255]

    # Count distinct blobs, not pixels (FIX)
    num_labels, _ = cv2.connectedComponents(mask)
    num_corners   = num_labels - 1      # subtract background label

    return corner_img, num_corners


def orb_features(image: np.ndarray, n_features: int = 500) -> tuple:
    """
    Stage 4b — ORB Feature Extraction (Oriented FAST + Rotated BRIEF).

    ORB is a fully classical binary feature descriptor:
      • Keypoint detection : FAST (Features from Accelerated Segment Test)
      • Orientation        : Intensity centroid moments (rotation-invariant)
      • Descriptor         : rBRIEF (Rotation-aware BRIEF)

    None-guard: detectAndCompute returns None for descriptors when the image
    has no detectable texture. Handled explicitly to prevent downstream crash.

    Args:
        image      : BGR restored image.
        n_features : Maximum number of features to retain.

    Returns:
        (feature_image, keypoints, descriptors)
        descriptors is an empty ndarray (not None) if no features found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb  = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # FIX: guard against None descriptor
    if descriptors is None:
        descriptors = np.array([], dtype=np.uint8)
        warnings.warn("ORB: no descriptors found — image may lack texture.")

    feature_img = cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return feature_img, keypoints, descriptors


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 – CANNY EDGE DETECTION  (bonus stage)
# ─────────────────────────────────────────────────────────────────────────────

def canny_edges(image: np.ndarray,
                low_threshold: int = 50,
                high_threshold: int = 150) -> np.ndarray:
    """
    Stage 5 — Canny Edge Detection.

    The Canny algorithm (Canny, 1986):
        1. Gaussian smoothing (internal noise suppression).
        2. Sobel gradient magnitude + direction.
        3. Non-maximum suppression → single-pixel-wide edges.
        4. Hysteresis thresholding → strong/weak edge linking.

    Applied to the restored image to extract structural edge maps.
    Returned as a BGR image (white edges on black) for grid insertion.

    Args:
        image          : BGR restored image.
        low_threshold  : Lower hysteresis bound.
        high_threshold : Upper hysteresis bound. Recommended ratio ≈ 1:2–1:3.

    Returns:
        BGR edge map.
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 – PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(image: np.ndarray,
                 sharpen_method: str = "unsharp",
                 clip_limit: float = 2.0,
                 tile_grid: tuple = (8, 8),
                 verbose: bool = True) -> dict:
    """
    Execute the full 5-stage pipeline on a single image.

    Stages:
        1. Preprocessing    → noise reduction  (Gaussian + Median)
        2. Enhancement      → local contrast   (CLAHE in LAB)
        3. Restoration      → detail recovery  (Unsharp Mask / Laplacian)
        4. Feature Analysis → corner/blob      (Harris + ORB)
        5. Edge Detection   → structural edges (Canny)

    Args:
        image          : Raw BGR image array.
        sharpen_method : 'unsharp' | 'laplacian'
        clip_limit     : CLAHE clip limit.
        tile_grid      : CLAHE tile grid dimensions.
        verbose        : Print per-stage timing to stdout.

    Returns:
        dict — original, preprocessed, enhanced, restored,
               harris, orb, edges, num_corners, num_keypoints,
               descriptors, total_ms
    """
    results = {"original": image.copy()}
    t0 = time.perf_counter()

    def _tick(label, start):
        if verbose:
            print(f"  {label:<40}: {(time.perf_counter()-start)*1000:6.1f} ms")

    t = time.perf_counter()
    results["preprocessed"] = preprocess(image)
    _tick("[Stage 1] Preprocessing", t)

    t = time.perf_counter()
    results["enhanced"] = apply_clahe(results["preprocessed"],
                                      clip_limit=clip_limit,
                                      tile_grid=tile_grid)
    _tick("[Stage 2] CLAHE Enhancement", t)

    t = time.perf_counter()
    results["restored"] = restore(results["enhanced"], method=sharpen_method)
    _tick(f"[Stage 3] Sharpening ({sharpen_method})", t)

    t = time.perf_counter()
    results["harris"], results["num_corners"] = harris_corners(results["restored"])
    _tick("[Stage 4a] Harris Corners", t)

    t = time.perf_counter()
    results["orb"], kps, results["descriptors"] = orb_features(results["restored"])
    results["num_keypoints"] = len(kps)
    _tick("[Stage 4b] ORB Features", t)

    t = time.perf_counter()
    results["edges"] = canny_edges(results["restored"])
    _tick("[Stage 5] Canny Edges", t)

    results["total_ms"] = (time.perf_counter() - t0) * 1000

    if verbose:
        print(f"  {'─'*48}")
        print(f"  {'Harris corner regions':<40}: {results['num_corners']}")
        print(f"  {'ORB keypoints':<40}: {results['num_keypoints']}")
        print(f"  {'Total pipeline time':<40}: {results['total_ms']:6.1f} ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 – QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB): PSNR = 20·log₁₀(255 / √MSE).

    Note: Since the pipeline intentionally alters intensity distribution
    (CLAHE, sharpening), a lower PSNR does not indicate quality loss — it
    reflects deliberate enhancement. Report alongside sharpness metrics.
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    No-reference sharpness metric via Laplacian variance.
    High variance ↔ sharp (high-frequency rich) image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 – VISUALISATION & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def build_histogram_panel(original: np.ndarray,
                          enhanced: np.ndarray,
                          width: int = 400,
                          height: int = 300) -> np.ndarray:
    """
    Draw a side-by-side normalised histogram comparison (before vs after CLAHE).

    Colour coding: Blue=B, Green=G, Red=R channel.
    Strong visual evidence of CLAHE's contrast stretching effect.
    """
    def draw_hist(img, title):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for ch, color in enumerate([(255, 80, 80), (80, 255, 80), (80, 80, 255)]):
            hist = cv2.calcHist([img], [ch], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, height - 44, cv2.NORM_MINMAX)
            pts = [(int(x * width / 256), height - 38 - int(y))
                   for x, y in enumerate(hist.flatten())]
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i+1], color, 1)
        cv2.line(canvas, (0, height-38), (width, height-38), (70, 70, 70), 1)
        cv2.putText(canvas, title, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "B  G  R", (8, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        return canvas

    return np.hstack([draw_hist(original, "Histogram: Original"),
                      draw_hist(enhanced,  "Histogram: Post-CLAHE")])


def build_comparison_grid(results: dict) -> np.ndarray:
    """
    Assemble a 3×3 visual comparison grid of all pipeline stages.

    Layout:
        Row 1: Original          | Preprocessed       | CLAHE Enhanced
        Row 2: Sharpened         | Harris Corners      | ORB Features
        Row 3: Canny Edges       | Histogram (spans 2 cols)
    """
    CELL_W, CELL_H = 400, 300

    panels = [
        ("Original",           results["original"]),
        ("1. Preprocessed",    results["preprocessed"]),
        ("2. CLAHE Enhanced",  results["enhanced"]),
        ("3. Sharpened",       results["restored"]),
        ("4a. Harris Corners", results["harris"]),
        ("4b. ORB Features",   results["orb"]),
        ("5. Canny Edges",     results["edges"]),
    ]

    cells = []
    for label, img in panels:
        cell = letterbox_resize(img, CELL_W, CELL_H)
        overlay = cell.copy()
        cv2.rectangle(overlay, (0, CELL_H - 32), (CELL_W, CELL_H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, cell, 0.4, 0, cell)
        cv2.putText(cell, label, (8, CELL_H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        cells.append(cell)

    hist_panel = build_histogram_panel(
        results["original"], results["enhanced"], width=CELL_W, height=CELL_H
    )

    row1 = np.hstack(cells[0:3])
    row2 = np.hstack(cells[3:6])
    row3 = np.hstack([cells[6], hist_panel])   # Canny + dual histogram
    return np.vstack([row1, row2, row3])


def add_metrics_bar(grid: np.ndarray, results: dict,
                    psnr: float, lap_orig: float, lap_proc: float) -> np.ndarray:
    """Append a single-line metrics summary bar beneath the grid."""
    bar   = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    delta = lap_proc - lap_orig
    sign  = "+" if delta >= 0 else ""
    pct   = f"{sign}{delta / max(lap_orig, 1) * 100:.1f}%"
    text  = (f"  PSNR: {psnr:.1f} dB (informational)  |  "
             f"Sharpness: {lap_orig:.0f}→{lap_proc:.0f} ({pct})  |  "
             f"Harris regions: {results['num_corners']}  |  "
             f"ORB kpts: {results['num_keypoints']}  |  "
             f"Total: {results['total_ms']:.0f} ms")
    cv2.putText(bar, text, (6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 215, 255), 1, cv2.LINE_AA)
    return np.vstack([grid, bar])


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9 – DEMO IMAGE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_image(size: int = 512) -> np.ndarray:
    """
    Synthetic test image for pipeline validation (no external file needed).

    Contains:
        • Geometric shapes (rectangles, circles, lines) — rich corners/edges
        • Diagonal intensity gradient
        • Moderate Gaussian noise (σ=10, realistic level — reduced from σ=18
          to avoid an artificially dramatic before/after effect)
        • Light salt-and-pepper noise (1% density)
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 40

    cv2.rectangle(img, (60,  60),  (200, 200), (180, 100,  60), -1)
    cv2.rectangle(img, (250, 80),  (420, 180), ( 60, 140, 200), -1)
    cv2.rectangle(img, (300, 300), (460, 460), (140,  60, 180), -1)
    cv2.circle(img, (150, 370), 70, (200, 200,  80), -1)
    cv2.circle(img, (370, 370), 55, ( 80, 200, 120), -1)
    cv2.line(img, (10,  10),       (size-10, size-10), (200, 80, 200), 2)
    cv2.line(img, (size-10, 10),   (10, size-10),      (200, 80,  80), 2)
    cv2.rectangle(img, (20, 20), (size-20, size-20), (100, 100, 100), 1)

    gradient = np.linspace(0, 35, size).astype(np.int16)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + gradient[np.newaxis, :], 0, 255)

    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    sp = np.random.random(img.shape[:2])
    img[sp < 0.010] = 0
    img[sp > 0.990] = 255

    return img


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10 – CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image_enhancement_pipeline",
        description=(
            "Advanced Image Enhancement & Feature Analysis Pipeline\n"
            "CSE3010 Computer Vision — Classical CV Techniques (Modules 1-4)\n"
            "Stages: Gaussian | Median | CLAHE | Unsharp/Laplacian | Harris | ORB | Canny\n"
            "Constraint: No deep learning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── DEMO MODE ──
    dm = sub.add_parser("demo", help="Run on a synthetic test image (no file needed)")
    dm.add_argument("--size",       type=int,   default=512)
    dm.add_argument("--sharpen",    choices=["unsharp", "laplacian"], default="unsharp")
    dm.add_argument("--clip",       type=float, default=2.0,
                    help="CLAHE clip limit (default: 2.0)")
    dm.add_argument("--tile",       type=int,   default=8,
                    help="CLAHE tile grid N×N (default: 8)")
    dm.add_argument("--output",     type=str,   default="demo_output.png")
    dm.add_argument("--no-display", action="store_true",
                    help="Skip cv2.imshow (headless/server environments)")

    # ── PROCESS MODE ──
    pm = sub.add_parser("process", help="Run on a user-supplied image file")
    pm.add_argument("input",        type=str)
    pm.add_argument("--sharpen",    choices=["unsharp", "laplacian"], default="unsharp")
    pm.add_argument("--clip",       type=float, default=2.0,
                    help="CLAHE clip limit (default: 2.0)")
    pm.add_argument("--tile",       type=int,   default=8,
                    help="CLAHE tile grid N×N (default: 8)")
    pm.add_argument("--output",     type=str,   default=None)
    pm.add_argument("--save-stages", action="store_true",
                    help="Save each pipeline stage as a separate PNG")
    pm.add_argument("--no-display", action="store_true")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Advanced Image Enhancement & Feature Analysis Pipeline  ║")
    print("║  CSE3010 Computer Vision  |  Classical CV (Modules 1-4)  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    tile_grid = (args.tile, args.tile)

    # ── Image acquisition ──
    if args.mode == "demo":
        print(f"[MODE]    demo  |  synthetic {args.size}×{args.size} px")
        image       = generate_demo_image(size=args.size)
        output_path = args.output
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"[ERROR] cv2.imread failed: {input_path}", file=sys.stderr)
            sys.exit(1)
        image = ensure_bgr(raw)
        print(f"[MODE]    process  |  {input_path.name}  "
              f"({image.shape[1]}×{image.shape[0]} px)")
        output_path = args.output or f"{input_path.stem}_enhanced.png"

    print(f"[SHARPEN] {args.sharpen}  |  CLAHE clip={args.clip}  tile={args.tile}×{args.tile}\n")
    print("Running pipeline:")
    print("─" * 52)

    # ── Run ──
    results = run_pipeline(image,
                           sharpen_method=args.sharpen,
                           clip_limit=args.clip,
                           tile_grid=tile_grid,
                           verbose=True)

    # ── Metrics ──
    psnr     = compute_psnr(results["original"], results["restored"])
    lap_orig = compute_laplacian_variance(results["original"])
    lap_proc = compute_laplacian_variance(results["restored"])
    pct      = (lap_proc - lap_orig) / max(lap_orig, 1) * 100

    print(f"\n  PSNR (orig vs restored)  : {psnr:.2f} dB")
    print(f"  [note: lower PSNR expected — CLAHE+sharpening intentionally shifts intensity]")
    print(f"  Sharpness (Lap. var)     : {lap_orig:.1f} → {lap_proc:.1f}  "
          f"({'+'if pct>=0 else ''}{pct:.1f}%)")

    # ── Optional per-stage saves ──
    if args.mode == "process" and args.save_stages:
        stem = input_path.stem
        for tag, img in [("preprocessed", results["preprocessed"]),
                         ("clahe",        results["enhanced"]),
                         ("sharpened",    results["restored"]),
                         ("harris",       results["harris"]),
                         ("orb",          results["orb"]),
                         ("canny",        results["edges"])]:
            p = f"{stem}_{tag}.png"
            cv2.imwrite(p, img)
            print(f"  Saved → {p}")

    # ── Grid ──
    grid = build_comparison_grid(results)
    grid = add_metrics_bar(grid, results, psnr, lap_orig, lap_proc)
    cv2.imwrite(output_path, grid)
    print(f"\n  ✔  Comparison grid saved → {output_path}")

    # ── Display ──
    if not args.no_display:
        try:
            cv2.imshow("Enhancement Pipeline — CSE3010", grid)
            print("  [Press any key to close the window]")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("  [INFO] cv2.imshow unavailable — "
                  "re-run with --no-display to suppress.")

    print("\n  Pipeline complete.\n")


if __name__ == "__main__":
    main()
