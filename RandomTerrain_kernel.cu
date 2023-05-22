#include "cuda_noise.cuh"

// Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b) { return (a + (b - 1)) / b; }

// update height map values
__global__ void generateHeightmapKernel(float *heightMap, unsigned int width, unsigned int seed) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x) * 16.0f;
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y) * 16.0f;

    float3 pos = make_float3(fx, fy, 0.0f);

    float tmp = cudaNoise::repeaterPerlin(pos, 0.2f, seed, 32, 1.8f, 0.45f);
    if (tmp < 0.01) {
        tmp = 0.01f;
    }
    heightMap[idx] = tmp;
}

// update height map values
__global__ void updateHeightmapKernel(float *heightMap, unsigned int width, unsigned int seed) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x) * 16.0f;
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y) * 16.0f;

    float3 pos = make_float3(fx, fy, 0.0f);

    float tmp = cudaNoise::repeaterPerlin(pos, 0.2f, seed, 32, 1.8f, 0.45f);
    if (tmp < 0.01) {
        tmp = 0.01f;
    }
    heightMap[idx] = tmp;
}

// generate slope by partial differences in spatial domain
__global__ void calculateSlopeKernel(float *h, float2 *slopeOut,
                                     unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * width + x;

    float2 slope = make_float2(0.0f, 0.0f);

    if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1)) {
        slope.x = h[i + 1] - h[i - 1];
        slope.y = h[i + width] - h[i - width];
    }

    slopeOut[i] = slope;
}

extern "C" void cudaGenerateHeightmapKernel(float *d_heightMap, unsigned int width, unsigned int height, unsigned int seed) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    generateHeightmapKernel<<<grid, block>>>(d_heightMap, width, seed);
}

extern "C" void cudaUpdateHeightmapKernel(float *d_heightMap, unsigned int width, unsigned int height, unsigned int seed) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    updateHeightmapKernel<<<grid, block>>>(d_heightMap, width, seed);
}

extern "C" void cudaCalculateSlopeKernel(float *hptr, float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    calculateSlopeKernel<<<grid, block>>>(hptr, slopeOut, width, height);
}
