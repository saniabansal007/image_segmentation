# CT Segmentation with Interactive Viewer

A modular C++/ITK/VTK/Qt6 pipeline implementing and comparing multiple segmentation
approaches on CT slice data, with a full-featured interactive Qt GUI viewer.

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| ITK | 5.x | Image I/O, filtering, segmentation algorithms |
| VTK | 9.x | 3-view orthogonal rendering, contour extraction |
| Qt6 | 6.x | GUI framework for the interactive viewer |
| CMake | >= 3.18 | Build system |
| GCC/Clang | C++17 | Compiler |

## Build (WSL Ubuntu / native Linux)

### 1. Install dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake build-essential \
    libinsighttoolkit5-dev \
    libvtk9-dev libvtk9-qt-dev \
    qt6-base-dev qt6-base-dev-tools \
    libqt6opengl6-dev libqt6openglwidgets6 \
    python3-pip

pip install tifffile   # fallback TIFF metadata sanitizer used by SliceIO
```

### 2. Configure and build

```bash
# From the project root
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This builds the following executables inside `build/`:

| Executable | What it does |
|---|---|
| `ct_preprocess` | **One-time Preprocessor**: extracts disk-backed axial, coronal, and sagittal slices with manifest |
| `test_lazy_loading` | **Verification Suite**: validates LRU cache bounds, concurrency, metadata, and slice correspondence |
| `ct_segmentation` | 2D: run one or all segmentation methods on a single slice |
| `ct_segmentation_3d` | 3D: stack a directory of slices into a volume and segment it |
| `ct_viewer` | **Interactive Qt GUI**: lazy multi-plane slice loading, crosshairs, window/level, ITK filters, ROI export |
| `debug_canny` | Developer tool to diagnose Canny threshold behaviour |
| `test_roi_processor` | Headless correctness test for ROI/region-growing/morphology logic |
| `ct_segmentation_gpu` | GPU-accelerated 3D median + morphology *(only built if CUDA compiler detected)* |

---

## Running

### Quick Start (One-Click Launcher) -- `./run.sh`

Launch the interactive viewer directly with automatic DISPLAY setup, compilation check, and dataset cache verification:

```bash
./run.sh
```

You can also pass custom paths or flags directly:
```bash
# Open a specific dataset or directory:
./run.sh data_preprocessed

# Open a single slice file:
./run.sh data/recon0434.tif
```

---

### 1. One-Time Preprocessing -- `ct_preprocess`

Run one-time preprocessing on your raw TIFF dataset to generate planar disk-backed raw slices (`axial/`, `coronal/`, `sagittal/`) with validation and `manifest.json`:

```bash
cd build

# Preprocess full dataset (generates data_preprocessed/ by default):
./ct_preprocess ../data -o ../data_preprocessed

# Test run on first 20 slices:
./ct_preprocess ../data -o ../data_preprocessed_test --max-slices 20

# Force regeneration:
./ct_preprocess ../data -o ../data_preprocessed --force
```

Subsequent launches detect the dataset fingerprint and reuse valid preprocessed data automatically.

### 2. Run Verification Tests -- `test_lazy_loading`

```bash
cd build
./test_lazy_loading
```

### 3. Interactive Viewer with Lazy Slice Loading -- `ct_viewer`

The viewer requires a live display (X11 or Wayland). In WSL, set up X11 forwarding first (see below).

```bash
cd build

# Open a preprocessed dataset (instant startup, lazy multi-planar loading):
./ct_viewer ../data_preprocessed

# Open a directory of TIFF slices (auto-detects preprocessed cache or loads on demand):
./ct_viewer ../data

# Open a single TIFF/PNG slice:
./ct_viewer ../data/recon0434.tif

# Auto-detect: looks for ./data_preprocessed, ./data, or ../data if no path given:
./ct_viewer
```

The Qt window opens with three synchronized orthogonal view tiles -- **Axial (Z)**, **Coronal (Y)**, **Sagittal (X)** -- plus a comprehensive control panel:

- **Dataset & Contrast (Window / Level)**:
  - Percentile-based Auto W/L (`Auto W/L` button) and custom Window/Level sliders with spinboxes.
  - Clinical Presets: Bone, Soft Tissue, Lung, Full Range, and Auto.
  - Interactive crosshairs toggle and one-click camera bounds reset.
- **Preprocessing Filters**:
  - Algorithms: Gaussian Smoothing, Median Filter, Anisotropic Diffusion, and Ring Artifact Correction.
  - **Scope Selection**: Choose between *Current Slices (All 3 Views - Fast)* for instant real-time feedback (< 100 ms) or *Full 3D Volume*.
  - Compare toggle (*Compare: Show Original CT*) for instant A/B inspection.
  - Fast filter reset preserving camera view, zoom, and current slice coordinates.
- **Segmentation Algorithms**:
  - 8 distinct methods: Threshold Range, Global Otsu, Multi-Otsu, Confidence-Connected Region Growing, Connected Threshold, K-Means Clustering, Canny Edge Morphology, and Watershed Segmentation.
  - **Scope Selection**: Instant multi-planar slice segmentation across all 3 views or complete 3D volume mask.
  - Choice of source image: run segmentation on either *Filtered Image* or *Original Image*.
- **ROI Boundary Contour & Translucent Fill**:
  - High-contrast vector contour outlines generated via `vtkContourFilter` (Marching Squares) with customizable line width (1.0–6.0 px) and high-visibility neon colors.
  - Translucent mask fill with real-time opacity slider (0–100%) and multi-color palettes.
  - Real-time legend bar for visual validation.
- **TIFF Export**:
  - Scope: Export current 2D axial slice or the entire 3D volume.
  - Content: Preprocessed/Filtered CT, Binary/Multi-label Segmentation Mask, or ROI-Masked CT volume.

#### Navigation & Interactions

- **Crosshair Navigation**: Left-click anywhere on any of the 3 orthogonal views to reposition the 3D crosshairs and synchronize the other two views instantly.
- **Zooming**: Scroll the mouse wheel inside any view tile to zoom in/out, or right-click drag up/down.
- **Panning**: Middle-click drag (or Shift + Left-click drag) to pan the image.
- **Slice Stepping**: Scroll mouse wheel with crosshair off or use the bottom navigation bar (◀ / ▶ buttons, direct spinbox entry, or draggable slider).
- **Smooth Sliders**: Sliders feature absolute click-to-position (clicking anywhere along the groove jumps directly to that position) and 60 FPS coalesced rendering without UI stutter.
- **Layout Switching**: Switch between 3-View Orthogonal, Axial Only, Coronal Only, and Sagittal Only via the View menu or the maximize button on each tile.

#### WSL X11 setup (Windows host)

Install an X server on Windows (e.g. [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
or [Xming](https://sourceforge.net/projects/xming/)), then in your WSL terminal:

```bash
export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=0   # use hardware OpenGL if available, else Mesa software
```

Add these two lines to your `~/.bashrc` so they apply automatically.

---

### 2D Segmentation CLI -- `ct_segmentation`

```bash
cd build
./ct_segmentation <slice.tif> [--method=name|all] [--classes=N] \
                   [--groundtruth=mask.png] [--ring-correct]
```

- `--method` -- one of the keys below, or `all` (default) to run every method
- `--classes` -- number of clusters/classes where applicable (default 3)
- `--groundtruth` -- label-mask PNG; prints Dice/Jaccard metrics per method
- `--ring-correct` -- apply ring-artifact suppression before segmenting

```bash
# Example: run all methods on a single slice with 4 classes
./ct_segmentation ../data/recon0434.tif --method=all --classes=4
```

Output PNGs are written next to the executable:
`output_<slice>_original.png`, `output_<slice>_preprocessed.png`,
`output_<slice>_<method>_segmented.png`

---

### 3D Segmentation CLI -- `ct_segmentation_3d`

```bash
cd build
./ct_segmentation_3d <slice_directory>
```

Stacks the `.tif` files in `<slice_directory>` (sorted by filename = scan order)
into a volume and runs 3D Multi-Otsu segmentation.

> **Memory note:** holds the raw volume + denoised copy + label volume simultaneously.
> A 900-slice 1000x1000 scan needs ~4-6 GB RAM. Use `ct_viewer` with `ct_preprocess` for memory-bounded on-demand viewing.

---

### Headless test hooks (for CI / no display)

```bash
# Take a one-frame screenshot and exit immediately:
./ct_viewer <data_dir> --screenshot output.png

# Seed, grow, and dilate before the screenshot:
./ct_viewer <data_dir> --test-roi <X> <Y> <Z> --screenshot output.png
```

---

## Project Structure

```
core/               ISegmenter interface + SegmenterFactory registry (2D)
io/                 TIFF/PNG I/O, incl. automatic malformed-metadata handling
preprocessing/      Denoising (Gaussian, Median, Anisotropic) + ring-artifact suppression
methods/            2D segmentation methods:
  threshold/        GlobalOtsu, MultiOtsu, AdaptiveThreshold
  region_growing/   ConfidenceConnected region growing
  clustering/       K-means
  edge_morphology/  Canny edges + morphological closing/fill
  watershed/        Gradient-magnitude watershed
evaluation/         Dice/Jaccard/overlap metrics vs. ground truth

volume/             3D: VolumeTypes, VolumeIO, SliceStore (LRU cache, multi-planar raw slices)
methods3d/          3D segmentation (MultiOtsuVolumeSegmenter)

viewer/
  OrthoSliceView.hpp/.cpp     Orthogonal VTK tile (coalesced 60 FPS rendering, crosshairs, Marching Squares)
  CTViewerWindow.hpp/.cpp     Full Qt main window (menus, control panel, 3-view layout,
                              preprocessing, segmentation, ROI export pipeline)
  ROIProcessor.hpp            Seed->crop->region-grow/morphology->TIFF (pure ITK, headless-safe)
  InteractiveViewerStyle.hpp  Legacy VTK-only interactor style

main.cpp            2D CLI driver
main_viewer.cpp     Qt interactive viewer entry point (ct_viewer)
main_volume.cpp     3D CLI driver (ct_segmentation_3d)
main_volume_gpu.cpp GPU-accelerated 3D driver (ct_segmentation_gpu)
tools/              ct_preprocess.cpp, test_lazy_loading.cpp, debug_canny.cpp, test_roi_processor.cpp
gpu/                CUDA kernels for MedianFilter3D, Morphology3D, GaussianFilter3D, and Segmentation3D
```

---

### GPU-Accelerated 3D Pipeline -- `ct_segmentation_gpu`

When compiled on a system with NVIDIA CUDA (e.g. RTX 4050 / RTX 30-series / 40-series), full 3D volumes can be filtered and segmented with native CUDA acceleration:

```bash
cd build
./ct_segmentation_gpu ../data_preprocessed
```

- **Median Denoising 3D**: Processes 847M voxels in parallel CUDA blocks with 26-neighbor shared memory windows.
- **Segmentation Algorithms on GPU**:
  - **Threshold Range 3D**: Direct parallel binary mask evaluation.
  - **Global Otsu 3D**: GPU atomic histogram construction + optimal between-class variance thresholding.
  - **Multi-Otsu 3D**: Full GPU multi-class intensity partitioning (2-4 classes).
  - **Connected Threshold 3D**: Iterative parallel wavefront expansion on 3D grid with atomic convergence tracking.
  - **Canny Morphology 3D**: GPU edge detection + parallel 3D morphological dilation/erosion/closing.
- **Interactive Viewer GPU Acceleration**: In `ct_viewer`, selecting **Full 3D Volume** mode now routes eligible segmentation algorithms directly to your NVIDIA GPU with seamless fallback to ITK CPU if CUDA is unavailable.

---

## Segmentation Methods

| Key | Method | Notes |
|---|---|---|
| `global_otsu` | Global binary Otsu thresholding | Fast baseline |
| `multi_otsu` | Multi-level Otsu (ITK native) | Good fit for distinct intensity bands |
| `adaptive_threshold` | Local/adaptive thresholding | Uses `itk::BoxMeanImageFilter` for speed |
| `region_growing` | Confidence-connected region growing | Seeds interactively via crosshair click |
| `connected_threshold` | Connected thresholding | Grew connected region within intensity range |
| `kmeans` | K-means clustering | Converges close to multi-Otsu on well-separated data |
| `canny_morphology` | Canny edges + morphological close/fill | Thresholds are percentile-based, not fixed |
| `watershed` | Gradient-magnitude watershed | Calibrated saliency, Otsu air suppression & 3 modes (Targeted, All, Boundaries) |

---

## Dataset Characteristics & Orthogonal Geometry Note

- **Source Slices**: The standard raw dataset (`data/`) consists of 847 TIFF files numbered `recon0061.tif` to `recon0907.tif` ($1000 \times 1000$ pixels each).
- **Z-Axis Truncation**: Slices `0001`–`0060` were omitted by the CT scanner operator. At the top of the scan (`recon0907.tif`), the physical specimen was taller than the detector height and continued upward. Consequently, Coronal and Sagittal vertical cross-sections exhibit a flat horizontal boundary at the top of the material where the scanner field of view ended.
- **Non-Cubic Dimensions**: Axial slices are $1000 \times 1000$ (square), while Coronal and Sagittal slices are $1000 \times 847$ (rectangular). All views are framed automatically with correct bounds.

---

## Bugs Found and Fixed

1. **TIFF resolution tag crash** -- Some slices have `XResolution`/`YResolution = 1000000/0`
   (zero denominator). `SliceIO::loadSlice` now catches this and retries against a
   metadata-sanitized copy written to the system temp directory.

2. **Canny thresholds were unit-agnostic** -- Fixed thresholds designed for 8-bit images were
   meaningless against 16-bit gradient magnitudes (median ~90, p99 ~1538). Now computed
   as **percentiles of the image's own gradient magnitude distribution** (lower=p90, upper=p99).

3. **Sanitized TIFF copies double-counted as extra slices** -- Fixed: (a) copies now go to
   the system temp directory; (b) `listSlices` skips any `*.cleaned.tif` names defensively.

4. **VTK Marching Squares 2D vs 3D Extent Error** -- Marching Squares (`vtkContourFilter`)
   failed with `Marching squares requires 2D data` when processing multi-plane 2D slices.
   Fixed by restricting `vtkExtractVOI` to the exact single-slice plane index for each view.

5. **QGroupBox Header Clipping & Border Strikethrough** -- Qt6 default stylesheets cut off
   group box headings on Linux and drew the frame border through text. Resolved with
   badge styling, explicit opaque backgrounds, and balanced margins.

6. **Slider Lag & Render Flooding** -- Dragging slice, window/level, or opacity sliders
   previously invoked up to 5 synchronous blocking OpenGL renders per event. Resolved by
   implementing `requestRender()` coalescing to lock rendering to 60 FPS without dropping frames,
   along with direct click-to-position slider groove tracking.

7. **Reset Filter Viewport Reset** -- Calling "Reset Filter" previously reinitialized geometry,
   wiped slice memory with zeros, and reset camera zoom and slice index. Fixed by refreshing
   only the 2D slice buffers while strictly preserving camera zoom, pan, and cursor position.

8. **Segmentation on 3D Volume Not Executing on GPU** -- 3D segmentation previously defaulted
   entirely to ITK single-threaded/slow CPU filters despite having an NVIDIA GPU. Implemented
   native CUDA 3D segmentation kernels (`gpu/Segmentation3D.cu`, `gpu/GPUSegmentation.hpp`)
   for Threshold Range, Global Otsu, Multi-Otsu, Connected Threshold, and Canny Morphology,
   achieving full 847M voxel segmentation in ~22 seconds on NVIDIA RTX GPUs.

9. **VTK OrthoSliceView Mask Fill Transparency Bug** -- The 2D slice mask fill actor lacked
   explicit `ColorWindow(1.0)` and `ColorLevel(0.5)`. Without this, VTK defaulted to
   window=255/level=127.5, causing binary label values 0 and 1 to map to index 0 (100% transparent),
   making the segmentation overlay completely invisible. Fixed in `viewer/OrthoSliceView.cpp`.

10. **Watershed Algorithm Flooded Viewport with Solid Yellow Mask** -- ITK's `WatershedImageFilter`
    partitions the entire domain into positive integer basin IDs with no concept of background air,
    labeling empty space outside the specimen. Furthermore, without gradient dynamic range normalization,
    raw CT gradients (up to 30,000+) rendered ITK's fractional `Threshold` and `Level` parameters
    ineffective, producing >140,000 micro-basins and marking 83.3% of the image as boundary lines.
    Fixed by:
    - Normalizing input intensities and Gaussian-smoothed gradient magnitudes to $[0.0, 1.0]$ via `itk::RescaleIntensityImageFilter`.
    - Automatically clamping dark air and image borders to 0 using Otsu foreground masking.
    - Providing 3 distinct modes: *Targeted Structure at Seed (Interactive)*, *All Segmented Structures*, and *Watershed Boundary Outlines*.
    - Exposing calibrated controls: Watershed Threshold (default 0.05), Level (default 0.20), and Sigma (1.5).

---

## Advanced Engineering Frontiers

- **Out-of-Core GPU Streaming with 3D Halo Exchange**: Enables processing multi-gigavoxel CT volumes exceeding physical GPU VRAM via asynchronous CUDA streams, pinned host memory, and overlapping 3D halo boundary exchanges.

---

## GPU Acceleration Architecture

The GPU pipeline is integrated into the CMake build system and compiled via `nvcc` targeting modern NVIDIA architectures (`sm_75`, `sm_80`, `sm_86`, `sm_89` — Turing, Ampere, Ada Lovelace):

- **Neighborhood Stencils on CUDA**: Memory-bound 3D neighborhood operations (median denoising, binary morphology, 3D Canny edges) are offloaded to CUDA blocks with shared-memory caching, eliminating the primary CPU performance bottleneck on 847M voxel scans.
- **Fast CPU Stays on CPU**: Single-pass histogram algorithms (e.g., Multi-Otsu) execute in sub-second time on CPU without incurring host-to-device memory copy overhead.
- **Interactive Routing**: In `ct_viewer`, selecting **Full 3D Volume** mode routes eligible algorithms directly to your GPU, falling back seamlessly to multi-threaded CPU ITK pipelines if CUDA is unavailable.
- **Benchmarking**: Compare GPU performance directly against the CPU reference using `./ct_segmentation_gpu --compare-cpu`.

## Evaluation

`evaluation/Evaluator.hpp` wraps `itk::LabelOverlapMeasuresImageFilter` to compute
Dice coefficient, Jaccard index, volume similarity, and false positive/negative error
against a ground-truth label PNG. Printed automatically when `--groundtruth` is passed.
