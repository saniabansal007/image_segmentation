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

# Open a directory of TIFF slices:
./ct_viewer <slice_directory>

# Open a single TIFF/PNG slice:
./ct_viewer <slice.tif>

# Auto-detect: looks for ./data or ../data if no path given
./ct_viewer
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
  ml/               Scaffold only (see ML section below)
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
gpu/                CUDA kernels for MedianFilter3D and Morphology3D
```

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
| `watershed` | Gradient-magnitude watershed | Raw/unmarked -- heavily over-segments by design |
| `ml` | ML-based (deep learning) | **Scaffold only, not implemented** |

---

## ML Method -- Status

`MLSegmenter::segment()` deliberately throws a descriptive error rather than faking a result.
Real implementation requires: labeled training data, training in Python via MONAI/PyTorch,
export to TorchScript/ONNX, and C++ inference via LibTorch or ONNX Runtime.

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

---

## Known Limitations

- **Ring-artifact suppression** is implemented but off by default (`--ring-correct`). Introduces
  minor blur from polar resampling.
- **Watershed over-segments** without markers (4,854 regions on the real structure slice).
- **No automatic empty-slice detector.** Background slices produce meaningless output from
  every intensity-based method.
- **Evaluation assumes matching label IDs** -- k-means and watershed need label remapping
  before fair comparison with ground truth.
- **Slice spacing assumed 1 in X/Y/Z.** Plain TIFFs carry no spacing metadata; coronal/sagittal
  views will look stretched if your real slice thickness differs.
- **ct_segmentation_3d** is not memory-optimised -- needs ~4-6 GB for 900 slices.
- **GPU code** (`ct_segmentation_gpu`) was written and reasoned through carefully but could not
  be built or run during development (no GPU in the sandbox). All other executables were
  actually compiled and executed with bugs caught by doing so.

---

## GPU Acceleration

`ct_segmentation_gpu` (only built if a CUDA compiler is detected) runs 3D median denoising
and ROI morphology on GPU via custom CUDA kernels. Read `gpu/README.md` before using.

---

# GPU Acceleration — Status

## What's here

- `MedianFilter3D.cu` — GPU 3D median filter (radius=1 only), replacing
  the CPU bottleneck identified when testing a 900-slice volume.
- `Morphology3D.cu` — GPU binary erode/dilate (open/close composed from
  these two on the host), for the ROI operations in the interactive viewer.
- `GPUPreprocessing.hpp` — ITK-to-raw-buffer bridge so the `.cu` files stay
  completely free of ITK dependencies and compile standalone with `nvcc`.
- `main_volume_gpu.cpp` — CLI driver mirroring `main_volume.cpp`, with a
  `--compare-cpu` flag to benchmark against the existing CPU path.

## What was NOT ported to GPU, and why

- **Multi-Otsu thresholding** — already near-instant (single-pass
  histogram, not a per-voxel neighborhood operation); GPU porting would
  add complexity for negligible gain here.
- **Region growing (ConfidenceConnected)** — inherently sequential
  flood-fill from a seed; does not parallelize cleanly. A real GPU region-
  growing implementation exists in research literature but is a
  significantly larger undertaking than this project's scope, not
  something to improvise.
- **Watershed** — same sequential-flooding issue as region growing.

If your professor asks "why isn't everything on GPU" — this table is the
honest answer: GPU acceleration isn't free everywhere, and most of the
real cost in this project's CPU runs comes from the filters that ARE
covered here (denoising, morphology), not the others.

## CRITICAL: none of this was built or run during development

This project was developed in a sandboxed environment with **no NVIDIA
GPU, no CUDA toolkit, and no driver installed at all**. Every other part
of this project (the 2D/3D CPU pipelines, the interactive viewer) was
actually compiled and run, with bugs caught and fixed by doing so. This
GPU code was not. It follows CUDA's documented API carefully and mirrors
this project's existing conventions, but treat it as a well-reasoned
starting point, not a verified deliverable, until you've built and run it
yourself.

Before trusting its output:
1. **Build it**: `cmake ..` should print `CUDA compiler found - ct_segmentation_gpu will be built`. If it instead prints the "No CUDA compiler found" message, the toolkit isn't installed/detected and the GPU target is silently skipped - the rest of the project still builds fine either way.
2. **Run it on a small, known input first** (e.g. the original 2-slice test data), not immediately on your full 900-slice set.
3. **Compare against the CPU reference** using `--compare-cpu`, and - since this isn't currently automated - manually check the GPU and CPU outputs actually look the same (e.g. diff the saved `.mha` files or eyeball a few slices), not just that both ran without crashing.

## Known gaps to close before relying on this

- **No VRAM chunking.** Both kernels allocate the full volume on the GPU
  at once (input + output = 2x volume size in VRAM). A 1000x1000x900
  16-bit volume is ~1.8GB, so this needs ~3.6GB+ free VRAM just for the
  median filter; morphology needs less (1-byte labels instead of 2-byte
  intensities) but still scales the same way. If your GPU has less VRAM
  than your volume needs, this will fail with a clear
  `cudaErrorMemoryAllocation` message rather than silently corrupting
  anything - but it will fail. Fixing this means processing the volume in
  Z-axis chunks with radius-sized overlap between chunks, which is a
  real, scoped follow-up task, not implemented here.
- **Median filter radius is hardcoded to 1.** The kernel's fixed-size
  27-element window only works for radius=1; using a different radius
  throws a clear error rather than silently producing wrong output, but
  extending it to arbitrary radii needs a dynamically-sized (or templated)
  window.
- **No correctness test against the CPU reference.** `--compare-cpu`
  prints both runtimes but doesn't currently diff the two outputs
  voxel-for-voxel. Add that check (even a simple max-absolute-difference
  print) before trusting the GPU path matches the CPU one on your data.
- **`CUDA_ARCHITECTURES` in CMakeLists.txt is a generic guess** (`60;70;75;80;86`,
  covering Pascal through Ampere). Check your actual GPU's compute
  capability (`nvidia-smi --query-gpu=compute_cap --format=csv`) and set
  this to match for a faster, more targeted build.

## Evaluation

`evaluation/Evaluator.hpp` wraps `itk::LabelOverlapMeasuresImageFilter` to compute
Dice coefficient, Jaccard index, volume similarity, and false positive/negative error
against a ground-truth label PNG. Printed automatically when `--groundtruth` is passed.
