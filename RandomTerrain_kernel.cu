#include "cuda_noise.cuh"

// Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b) { return (a + (b - 1)) / b; }

// update height map values
__global__ void
generateHeightmapKernel(float *heightMap, float *heightMapPrev, unsigned int width, unsigned int height, int seed) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    if (heightMapPrev != nullptr && y == 0) {
        heightMap[idx] = heightMapPrev[(height - 1) * width + x];
        return;
    }

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x) * 16.0f;
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y) * 16.0f;

    float3 pos = make_float3(fx, fy, 0.0f);

    float tmp = cudaNoise::repeaterPerlin(pos, 0.2f, seed, 32, 1.8f, 0.45f);
    if (tmp < 0.01) {
        tmp = 0.01f;
    }
    heightMap[idx] = tmp;
}

__global__ void copyOverKernel(float *heightMap, float *heightMapPrev, unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    if (y == 0) {
        heightMap[idx] = heightMapPrev[(height - 1) * width + x];
    }
}

__global__ void perlinKernel(float *heightMap, unsigned int width, unsigned int height, int seed) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    if (y == height - 1) {
        return;
    }

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
__global__ void updateHeightmapKernel(
        float *heightMap,
        float *heightMapNext,
        float *heightMapOut,
        unsigned int width,
        unsigned int height,
        unsigned int rowOffset) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = y * width + x;

    if (y < rowOffset) {
        unsigned int row = (height - rowOffset) + y;
        heightMapOut[idx] = heightMapNext[row * width + x];
    } else {
        unsigned int row = y - rowOffset;
        heightMapOut[idx] = heightMap[row * width + x];
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

extern "C" void
cudaGenerateHeightmapKernel(float *d_heightMap, float *d_heightMapPrev, unsigned int width, unsigned int height,
                            int seed) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    generateHeightmapKernel<<<grid, block>>>(d_heightMap, d_heightMapPrev, width, height, seed);
}

extern "C" void
cudaCopyOverKernel(float *d_heightMap, float *d_heightMapPrev, unsigned int width, unsigned int height) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    copyOverKernel<<<grid, block>>>(d_heightMap, d_heightMapPrev, width, height);
}

extern "C" void
cudaPerlinKernel(float *d_heightMap, unsigned int width, unsigned int height, int seed) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    perlinKernel<<<grid, block>>>(d_heightMap, width, height, seed);
}

extern "C" void
cudaUpdateHeightmapKernel(float *d_heightMap, float *d_heightMapNext, float *heightMapOut, unsigned int width,
                          unsigned int height, unsigned int rowOffset) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    updateHeightmapKernel<<<grid, block>>>(d_heightMap, d_heightMapNext, heightMapOut, width, height, rowOffset);
}

extern "C" void cudaCalculateSlopeKernel(float *hptr, float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height) {
    dim3 block(8, 8, 1);
    dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    calculateSlopeKernel<<<grid, block>>>(hptr, slopeOut, width, height);
}
