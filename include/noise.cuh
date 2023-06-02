#include <cstdio>
#include <cmath>
#include <ctime>
#include "helper_cuda.h"

class PerlinGenerator {
private:
    int seed;

    __device__
    int hash(int x, int y) const {
        static int hash_values[] = {
                208, 34, 231, 213, 32, 248, 233, 56,
                161, 78, 24, 140, 71, 48, 140, 254,
                245, 255, 247, 247, 40, 185, 248, 251,
                245, 28, 124, 204, 204, 76, 36, 1,
                107, 28, 234, 163, 202, 224, 245, 128,
                167, 204, 9, 92, 217, 54, 239, 174,
                173, 102, 193, 189, 190, 121, 100, 108,
                167, 44, 43, 77, 180, 204, 8, 81,
                70, 223, 11, 38, 24, 254, 210, 210,
                177, 32, 81, 195, 243, 125, 8, 169,
                112, 32, 97, 53, 195, 13, 203, 9,
                47, 104, 125, 117, 114, 124, 165, 203,
                181, 235, 193, 206, 70, 180, 174, 0,
                167, 181, 41, 164, 30, 116, 127, 198,
                245, 146, 87, 224, 149, 206, 57, 4,
                192, 210, 65, 210, 129, 240, 178, 105,
                228, 108, 245, 148, 140, 40, 35, 195,
                38, 58, 65, 207, 215, 253, 65, 85,
                208, 76, 62, 3, 237, 55, 89, 232,
                50, 217, 64, 244, 157, 199, 121, 252,
                90, 17, 212, 203, 149, 152, 140, 187,
                234, 177, 73, 174, 193, 100, 192, 143,
                97, 53, 145, 135, 19, 103, 13, 90,
                135, 151, 199, 91, 239, 247, 33, 39,
                145, 101, 120, 99, 3, 186, 86, 99,
                41, 237, 203, 111, 79, 220, 135, 158,
                42, 30, 154, 120, 67, 87, 167, 135,
                176, 183, 191, 253, 115, 184, 21, 233,
                58, 129, 233, 142, 39, 128, 211, 118,
                137, 139, 255, 114, 20, 218, 113, 154,
                27, 127, 246, 250, 1, 8, 198, 250,
                209, 92, 222, 173, 21, 88, 102, 219,
        };

        int tmp = hash_values[(y + seed) % 256];
        return hash_values[(tmp + x) % 256];
    }

    static __device__
    float linear_interpolation(float x, float y, float s) {
        return x + s * (y - x);
    }

    static __device__
    float2 gradient(int val) {
        switch (val % 4) {
            case 0:
                return make_float2(1.0, 1.0);
            case 1:
                return make_float2(-1.0, 1.0);
            case 2:
                return make_float2(-1.0, -1.0);
            case 3:
                return make_float2(1.0, -1.0);
        }
    }

    static __device__
    float dot(float2 a, float2 b) {
        return a.x * b.x + a.y * b.y;
    }

    // Ken Perlin's fade function for Perlin noise
    static __device__
    float fade(float t) {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // 6t^5 - 15t^4 + 10t^3
    }

public:
    explicit __device__ PerlinGenerator(int seed) : seed(seed) {}


    __device__
    float perlin2d(float2 coords) const {
        int x_int = static_cast<int>(floorf(coords.x)) % 255;
        int y_int = static_cast<int>(floorf(coords.y)) % 255;

        float x_rem = coords.x - floorf(coords.x);
        float y_rem = coords.y - floorf(coords.y);

        // b = bottom, t = top, r = right, l = left

        float2 tr = make_float2(x_rem - 1.0, y_rem - 1.0);
        float2 tl = make_float2(x_rem, y_rem - 1.0);
        float2 br = make_float2(x_rem - 1.0, y_rem);
        float2 bl = make_float2(x_rem, y_rem);

        int htr = hash(x_int + 1, y_int + 1);
        int htl = hash(x_int, y_int + 1);
        int hbr = hash(x_int + 1, y_int);
        int hbl = hash(x_int, y_int);

        float vtr = dot(tr, gradient(htr));
        float vtl = dot(tl, gradient(htl));
        float vbr = dot(br, gradient(hbr));
        float vbl = dot(bl, gradient(hbl));

        float u = fade(x_rem);
        float v = fade(y_rem);

        float interpolation_left = linear_interpolation(vbl, vtl, v);
        float interpolation_right = linear_interpolation(vbr, vtr, v);
        return linear_interpolation(interpolation_left, interpolation_right, u);
    }

    __device__
    float fbm(float2 coords, float freq, float lacunarity, float decay, int octaves) const {
        float res = 0.0;
        float amp = 1.0;
        for (int i = 0; i < octaves; i++) {
            res += amp * perlin2d(make_float2(coords.x * freq, coords.y * freq));
            amp *= decay;
            freq *= lacunarity;
        }

        return res * 0.5f + 0.5f; // from [-1, 1] to [0, 1]
    }
};