#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            std::ostringstream oss__;                                                       \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "              \
                  << cudaGetErrorString(err__);                                             \
            throw std::runtime_error(oss__.str());                                          \
        }                                                                                   \
    } while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

static std::string nextToken(std::ifstream& in) {
    std::string token;
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        return token;
    }
    return {};
}

Image loadPGM(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input: " + path.string());

    const std::string magic = nextToken(in);
    if (magic != "P5") {
        throw std::runtime_error("Only binary PGM (P5) is supported: " + path.string());
    }

    const std::string w = nextToken(in);
    const std::string h = nextToken(in);
    const std::string maxv = nextToken(in);
    if (w.empty() || h.empty() || maxv.empty()) {
        throw std::runtime_error("Invalid PGM header: " + path.string());
    }

    Image img;
    img.width = std::stoi(w);
    img.height = std::stoi(h);
    const int maxValue = std::stoi(maxv);
    if (img.width <= 0 || img.height <= 0 || maxValue != 255) {
        throw std::runtime_error("PGM must be 8-bit with max value 255: " + path.string());
    }

    in.get(); // consume one whitespace byte after the max value
    img.pixels.resize(static_cast<size_t>(img.width) * img.height);
    in.read(reinterpret_cast<char*>(img.pixels.data()), static_cast<std::streamsize>(img.pixels.size()));
    if (in.gcount() != static_cast<std::streamsize>(img.pixels.size())) {
        throw std::runtime_error("Unexpected end of PGM pixel data: " + path.string());
    }
    return img;
}

void savePGM(const fs::path& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create output: " + path.string());
    out << "P5\n" << img.width << ' ' << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.pixels.data()),
              static_cast<std::streamsize>(img.pixels.size()));
}

__global__ void sobelKernel(const uint8_t* input, uint8_t* output, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    const int p00 = input[(y - 1) * width + (x - 1)];
    const int p01 = input[(y - 1) * width + x];
    const int p02 = input[(y - 1) * width + (x + 1)];
    const int p10 = input[y * width + (x - 1)];
    const int p12 = input[y * width + (x + 1)];
    const int p20 = input[(y + 1) * width + (x - 1)];
    const int p21 = input[(y + 1) * width + x];
    const int p22 = input[(y + 1) * width + (x + 1)];

    const int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
    const int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
    const float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
    output[y * width + x] = static_cast<uint8_t>(fminf(255.0f, magnitude));
}

struct Options {
    fs::path inputDir = "data/input";
    fs::path outputDir = "data/output";
};

Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) opt.inputDir = argv[++i];
        else if (arg == "--output" && i + 1 < argc) opt.outputDir = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./sobel_batch [--input DIR] [--output DIR]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return opt;
}

int main(int argc, char** argv) {
    try {
        const Options opt = parseArgs(argc, argv);
        fs::create_directories(opt.outputDir);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        std::vector<fs::path> files;
        if (!fs::exists(opt.inputDir)) {
            throw std::runtime_error("Input directory does not exist: " + opt.inputDir.string());
        }
        for (const auto& entry : fs::directory_iterator(opt.inputDir)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (ext == ".pgm") files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) throw std::runtime_error("No .pgm files found in " + opt.inputDir.string());

        size_t processed = 0;
        double totalGpuMs = 0.0;
        double totalCpuReadMs = 0.0;
        double totalCpuWriteMs = 0.0;

        std::cout << "CUDA GPU: " << prop.name << "\n";
        std::cout << "Input directory: " << opt.inputDir << "\n";
        std::cout << "Output directory: " << opt.outputDir << "\n";
        std::cout << "Images found: " << files.size() << "\n";

        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (const auto& path : files) {
            const auto readStart = std::chrono::steady_clock::now();
            Image img = loadPGM(path);
            const auto readEnd = std::chrono::steady_clock::now();
            totalCpuReadMs += std::chrono::duration<double, std::milli>(readEnd - readStart).count();

            const size_t bytes = img.pixels.size();
            uint8_t* d_input = nullptr;
            uint8_t* d_output = nullptr;
            CUDA_CHECK(cudaMalloc(&d_input, bytes));
            CUDA_CHECK(cudaMalloc(&d_output, bytes));
            CUDA_CHECK(cudaMemcpy(d_input, img.pixels.data(), bytes, cudaMemcpyHostToDevice));

            dim3 block(16, 16);
            dim3 grid((img.width + block.x - 1) / block.x,
                      (img.height + block.y - 1) / block.y);

            CUDA_CHECK(cudaEventRecord(start));
            sobelKernel<<<grid, block>>>(d_input, d_output, img.width, img.height);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float gpuMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
            totalGpuMs += gpuMs;

            Image result = img;
            CUDA_CHECK(cudaMemcpy(result.pixels.data(), d_output, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_output));

            const auto writeStart = std::chrono::steady_clock::now();
            const fs::path outPath = opt.outputDir / (path.stem().string() + "_sobel.pgm");
            savePGM(outPath, result);
            const auto writeEnd = std::chrono::steady_clock::now();
            totalCpuWriteMs += std::chrono::duration<double, std::milli>(writeEnd - writeStart).count();

            ++processed;
            if (processed % 10 == 0 || processed == files.size()) {
                std::cout << "Processed " << processed << "/" << files.size() << " images\n";
            }
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== RESULTS ===\n";
        std::cout << "Processed images: " << processed << "\n";
        std::cout << "Total GPU kernel time (ms): " << totalGpuMs << "\n";
        std::cout << "Average GPU kernel time/image (ms): " << (totalGpuMs / processed) << "\n";
        std::cout << "Total CPU read time (ms): " << totalCpuReadMs << "\n";
        std::cout << "Total CPU write time (ms): " << totalCpuWriteMs << "\n";
        std::cout << "GPU computation completed successfully.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
