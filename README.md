# stiching

[English](README.md) | [简体中文](README.zh-CN.md)

An experimental C++ image stitching project that integrates SuperPoint feature extraction and LightGlue feature matching into OpenCV Stitcher, using ONNX Runtime for neural network inference.

> The repository name remains `stiching`; the generated executable is named `mapstitch`.

## Features

- Extract image keypoints and descriptors with SuperPoint
- Match features with LightGlue
- Support for OpenCV `PANORAMA` and `SCANS` stitching modes
- Output stitched images
- Output matched point coordinates and match visualizations
- Optionally split input images into overlapping regions to increase opportunities for experimental matching

## Dependencies

- A compiler with C++17 support
- CMake 3.10+
- OpenCV
- ONNX Runtime
- CGAL
- GMP
- FFTW3
- SuperPoint and LightGlue ONNX models

The current `CMakeLists.txt` looks for ONNX Runtime at the following location by default:

```text
/opt/onnxruntime
```

If it is installed elsewhere, update `ONNXRUNTIME_ROOT`.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

The generated executable is usually located at:

```text
build/mapstitch
```

## Usage

```bash
./build/mapstitch \
  --sp ./models/superpoint.onnx \
  --lg ./models/lightglue.onnx \
  --mthresh 0.2 \
  --mode scans \
  --output result.jpg \
  --match matches.txt \
  image1.jpg image2.jpg
```

Main arguments:

```text
--sp <path>             SuperPoint ONNX model
--lg <path>             LightGlue ONNX model
--mthresh <float>       Matching threshold
--mode panorama|scans   OpenCV Stitcher mode
--output <path>         Stitched output file
--match <path>          Output file for matched point coordinates
--d3                    Split each image into multiple overlapping regions
```

View the built-in help:

```bash
./build/mapstitch --help
```

## Output

- The stitched image is written to the path specified by `--output`.
- Matched points are written to the file specified by `--match`.
- The code also generates and displays match visualizations. Running on a server without a graphical interface may require removing `imshow` / `waitKey` or using a virtual display.

## Current limitations

- The ONNX Runtime path is hard-coded in CMake.
- Model input and output formats must match the repository's SuperPoint/LightGlue wrappers.
- Some default output paths and arguments lack complete validation.
- The code still includes experimental modules and debugging displays; it is not a headless batch processing tool.
- Stitching quality depends on image overlap, viewpoint changes, model weights, and the matching threshold.
