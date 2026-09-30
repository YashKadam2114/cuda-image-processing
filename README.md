# CUDA Batch Image Processing Project

## Project title
GPU-Accelerated Batch Sobel Edge Detection using CUDA

## Objective
This project processes a large number of small grayscale images using a custom CUDA kernel. For every input image, the GPU applies a 3x3 Sobel edge-detection filter and writes the edge map to an output PGM image.

The project satisfies the assignment requirement for GPU computation because the core image-processing operation is implemented in `__global__` CUDA kernel code and executed with CUDA thread blocks on the GPU.

## Dataset
The included generator creates 200 small 256x256 PGM images. This demonstrates the required "100s of small inputs" workload without depending on an external download.

The generator is only for creating test data. The image processing itself is performed by CUDA.

## GPU computation
The kernel `sobelKernel` computes the horizontal and vertical Sobel gradients (`Gx` and `Gy`) independently for every pixel. Each CUDA thread processes one pixel.

Key CUDA API calls used:
- `cudaMalloc`
- `cudaMemcpy`
- CUDA kernel launch (`sobelKernel<<<grid, block>>>`)
- `cudaEventRecord` / `cudaEventElapsedTime`
- `cudaDevice` / `cudaGetDeviceProperties`

## Build and run
Run these commands from the repository root in the CUDA lab environment:

```bash
python3 scripts/generate_dataset.py --count 200 --width 256 --height 256
nvcc -O2 -std=c++17 src/sobel_batch.cu -o sobel_batch
./sobel_batch --input data/input --output data/output | tee proof/run.log
```

Expected output includes:

```text
CUDA GPU: <your GPU name>
Images found: 200
Processed 200/200 images
=== RESULTS ===
Processed images: 200
Total GPU kernel time (ms): ...
Average GPU kernel time/image (ms): ...
GPU computation completed successfully.
```

## Proof for peer grading
Commit these items to the public repository:

1. `src/sobel_batch.cu` — CUDA implementation.
2. `scripts/generate_dataset.py` — dataset generator.
3. `proof/run.log` — terminal output proving CUDA execution and 200 images were processed.
4. A few representative files from `data/output/` (for example 5–10 edge images). The full generated dataset does not need to be stored if repository size is a concern; the generator recreates it.
5. This README.

Take a screenshot showing the terminal with the GPU name, `Images found: 200`, successful processing, and the final GPU timing.

## Short description to submit
This project implements GPU-accelerated batch image processing using CUDA. A custom CUDA Sobel edge-detection kernel processes 200 small 256x256 grayscale images. Each CUDA thread computes the gradient magnitude for one pixel, enabling parallel image processing across the dataset. CUDA memory allocation, host-device transfers, kernel execution, and CUDA event timing are used to demonstrate real GPU computation. The output consists of Sobel edge maps for every input image, and the execution log records the GPU device, number of processed images, and GPU kernel timing.

## Notes
- The program accepts binary PGM (`P5`) images.
- If you use your own image dataset in PNG/JPG/JPEG format, convert it to 8-bit PGM before running the CUDA program, or extend `loadPGM` with another image decoder.
- CPU file I/O is expected; the grading requirement is that the actual image-processing computation runs on the GPU.
