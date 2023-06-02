#include "noise.cuh"
#include <cassert>

int idiv_ceil(int a, int b) { return (a + (b - 1)) / b; }

// update height map values
__global__ void
generateHeightmapKernel(float *heightMap, unsigned int width, int seed, unsigned int rowOffset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    PerlinGenerator gen(seed);
    float2 virtual_pos = make_float2(x, y + rowOffset);

    float val = gen.fbm(virtual_pos, 0.005, 2.0, 0.5, 16);

    heightMap[idx] = val > 0.3 ? val : 0.299;
}

// update height map values
__global__ void updateHeightmapKernel(
        const float *heightMap,
        const float *heightMapNext,
        float *heightMapOut,
        unsigned int width,
        unsigned int height,
        unsigned int rowOffset) {

    assert(rowOffset < height);

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    if (y + rowOffset < height) {
        unsigned int row = y + rowOffset;
        heightMapOut[idx] = heightMap[row * width + x];
    } else {
        unsigned int row = (y + rowOffset) - height;
        heightMapOut[idx] = heightMapNext[row * width + x];
    }
}

// generate slope by partial differences in spatial domain
__global__ void calculateSlopeKernel(float *h, float2 *slopeOut,
                                     unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    float2 slope = make_float2(0.0f, 0.0f);

    if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1)) {
        slope.x = h[idx + 1] - h[idx - 1];
        slope.y = h[idx + width] - h[idx - width];
    }

    slopeOut[idx] = slope;
}

__global__ void
getTerrainHeightKernel(const float *heightMap, float *out, unsigned int width, unsigned int height, int x, int y) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < height);

    *out = heightMap[y * width + x];
}

extern "C" void
cudaGenerateHeightmapKernel(float *d_heightMap, unsigned int width, unsigned int height,
                            int seed, unsigned int rowOffset) {
    dim3 block(8, 8, 1);
    dim3 grid(idiv_ceil(width, block.x), idiv_ceil(height, block.y), 1);
    generateHeightmapKernel<<<grid, block>>>(d_heightMap, width, seed, rowOffset);
}

extern "C" void
cudaUpdateHeightmapKernel(float *d_heightMap, float *d_heightMapNext, float *heightMapOut, unsigned int width,
                          unsigned int height, unsigned int rowOffset) {
    dim3 block(8, 8, 1);
    dim3 grid(idiv_ceil(width, block.x), idiv_ceil(height, block.y), 1);
    updateHeightmapKernel<<<grid, block>>>(d_heightMap, d_heightMapNext, heightMapOut, width, height, rowOffset);
}

extern "C" void cudaCalculateSlopeKernel(float *hptr, float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height) {
    dim3 block(8, 8, 1);
    dim3 grid(idiv_ceil(width, block.x), idiv_ceil(height, block.y), 1);
    calculateSlopeKernel<<<grid, block>>>(hptr, slopeOut, width, height);
}

extern "C" void
cudaGetTerrainHeightKernel(const float *d_heightMap, float *out, unsigned int width, unsigned int height, int x,
                           int y) {
    getTerrainHeightKernel<<<1, 1>>>(d_heightMap, out, width, height, x, y);
}