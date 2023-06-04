// includes
#include <helper_gl.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "util.h"

#include <GL/freeglut.h>

const char *programDesc = "CUDA Random Terrain Generator";

#define REFRESH_DELAY 10 // ms

////////////////////////////////////////////////////////////////////////////////
// constants
unsigned int windowW = 1600;
unsigned int windowH = 900;

const unsigned int meshSize = 256;

// OpenGL vertex buffers
GLuint posVertexBuffer;
GLuint heightVertexBuffer;
GLuint slopeVertexBuffer;
struct cudaGraphicsResource *cuda_heightVB_resource;
struct cudaGraphicsResource *cuda_slopeVB_resource;

GLuint indexBuffer;
GLuint shaderProg;
char *vertShaderPath = nullptr;
char *fragShaderPath = nullptr;

// mouse controls
int mouseOldX;
int mouseOldY;
int mouseButtons = 0;

struct CameraInfo cam = {
        20.0,
        180.0,
        0.0,
        0.0,
        -3.0
};
struct CameraInfo oldCam = cam;
const struct CameraInfo followCam = {
        30.0,
        180.0,
        0.0,
        -0.5,
        -1.5
};


bool animate = false;
bool followMode = false;
bool cameraLocked = false;

int seed = 0;
unsigned int step = 0;
int followTriggeredAtStep = -1;

float *d_heightMap = nullptr;
float *d_heightMapNext = nullptr;
float2 *d_slope = nullptr;
float *g_height = nullptr;

// pointers to device object
float *g_hptr = nullptr;
float2 *g_sptr = nullptr;

StopWatchInterface *timer = nullptr;
float animTime = 0.0f;
float prevTime = 0.0f;
float animationRate = -0.001f;


extern "C" void
cudaGenerateHeightmapKernel(float *d_heightMap, unsigned int width, unsigned int height,
                            int seed, unsigned int rowOffset);

extern "C" void
cudaUpdateHeightmapKernel(float *d_heightMap, float *d_heightMapNext, float *heightMapOut, unsigned int width,
                          unsigned int height, unsigned int rowOffset);

extern "C" void cudaCalculateSlopeKernel(float *h, float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height);
extern "C" void
cudaGetTerrainHeightKernel(const float *d_heightMap, float *out, unsigned int width, unsigned int height, int x,
                           int y);

////////////////////////////////////////////////////////////////////////////////
// forward declarations
void runTerrainGen(int argc, char **argv);

// GL functionality
bool initGL(int *argc, char **argv);

void createVBO(GLuint *vbo, int size);

void deleteVBO(GLuint *vbo);

void createMeshIndexBuffer(GLuint *id, int w, int h);

void createMeshPositionVBO(GLuint *id, int w, int h);

GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);

// rendering callbacks
void display();

void keyboard(unsigned char key, int x, int y);

void mouse(int button, int state, int x, int y);

void motion(int x, int y);

void reshape(int w, int h);

void timerEvent(int value);

// Cuda functionality
void runCuda();

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    printf(
            "[%s]\n\n"
            "Left mouse button          - rotate\n"
            "Middle mouse button        - pan\n"
            "Right mouse button         - zoom\n",
            programDesc);

    srand(static_cast<unsigned int>(time(nullptr)));
    runTerrainGen(argc, argv);
    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run terrain generator
////////////////////////////////////////////////////////////////////////////////
void runTerrainGen(int argc, char **argv) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("[%s] ", programDesc);
    printf("\n");

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA
    // interop.
    if (!initGL(&argc, argv)) {
        return;
    }

    findCudaDevice(argc, (const char **) (argv));

    constexpr int htSize = meshSize * meshSize * sizeof(float);
    constexpr int slopeSize = meshSize * meshSize * sizeof(float2);
    checkCudaErrors(cudaMalloc((void **) &d_heightMap, htSize));
    checkCudaErrors(cudaMalloc((void **) &d_heightMapNext, htSize));
    checkCudaErrors(cudaMalloc((void **) &d_slope, slopeSize));
    checkCudaErrors(cudaMallocHost((void **) &g_height, sizeof(float)));

    seed = rand();
    printf("Generating terrain with seed: %u\n", seed);
    cudaGenerateHeightmapKernel(d_heightMapNext, meshSize, meshSize, seed, 0);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    prevTime = sdkGetTimerValue(&timer);

    // create vertex buffers and register with CUDA
    createVBO(&heightVertexBuffer, htSize);
    checkCudaErrors(
            cudaGraphicsGLRegisterBuffer(&cuda_heightVB_resource, heightVertexBuffer,
                                         cudaGraphicsMapFlagsWriteDiscard));

    createVBO(&slopeVertexBuffer, slopeSize);
    checkCudaErrors(
            cudaGraphicsGLRegisterBuffer(&cuda_slopeVB_resource, slopeVertexBuffer,
                                         cudaGraphicsMapFlagsWriteDiscard));

    // create vertex and index buffer for mesh
    createMeshPositionVBO(&posVertexBuffer, meshSize, meshSize);
    createMeshIndexBuffer(&indexBuffer, meshSize, meshSize);

    runCuda();

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda kernels
////////////////////////////////////////////////////////////////////////////////
void runCuda() {
    size_t num_bytes;

    // update heightmap values in vertex buffer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_heightVB_resource, cudaStreamDefault));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            (void **) &g_hptr, &num_bytes, cuda_heightVB_resource));


    auto offset = step % meshSize;
    if (offset == 0) {
        cudaMemcpy(d_heightMap, d_heightMapNext, meshSize * meshSize * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaGenerateHeightmapKernel(d_heightMapNext, meshSize, meshSize, seed, step + meshSize);
    }

    cudaUpdateHeightmapKernel(
            d_heightMap,
            d_heightMapNext,
            g_hptr,
            meshSize,
            meshSize,
            offset
    );

    if (followMode) {
        cudaGetTerrainHeightKernel(g_hptr, g_height, meshSize, meshSize, (meshSize / 2), 0);
    }

    step++;

    // calculate slope for shading
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_slopeVB_resource, cudaStreamDefault));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            (void **) &g_sptr, &num_bytes, cuda_slopeVB_resource));

    cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_heightVB_resource, cudaStreamDefault));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_slopeVB_resource, cudaStreamDefault));
}

inline void adjustCamera() {
    if (followMode) {
        if (followTriggeredAtStep + 100 > step) { // moving to follow camera pos over 100 steps
            float xdiff = (followCam.translateX - oldCam.translateX) / 100.0f;
            float ydiff = (followCam.translateY - oldCam.translateY) / 100.0f;
            float zdiff = (followCam.translateZ - oldCam.translateZ) / 100.0f;
            float axdiff = fmod(followCam.rotateX - oldCam.rotateX, 360.0f) / 100.0f;
            float aydiff = fmod(followCam.rotateY - oldCam.rotateY, 360.0f) / 100.0f;

            cam.translateX += xdiff;
            cam.translateY += ydiff;
            cam.translateZ += zdiff;
            cam.rotateX += axdiff;
            cam.rotateY += aydiff;
        } else {
            cam = followCam;
            cam.translateY = -0.1f - *g_height;
        }
    } else {
        if (followTriggeredAtStep >= 0 && followTriggeredAtStep + 100 > step) { // moving to previous camera pos over 100 steps
            float xdiff = (oldCam.translateX - followCam.translateX) / 100.0f;
            float ydiff = (oldCam.translateY - followCam.translateY) / 100.0f;
            float zdiff = (oldCam.translateZ - followCam.translateZ) / 100.0f;
            float axdiff = fmod(oldCam.rotateX - followCam.rotateX, 360.0f) / 100.0f;
            float aydiff = fmod(oldCam.rotateY - followCam.rotateY, 360.0f) / 100.0f;

            cam.translateX += xdiff;
            cam.translateY += ydiff;
            cam.translateZ += zdiff;
            cam.rotateX += axdiff;
            cam.rotateY += aydiff;
        }
        if (followTriggeredAtStep + 100 <= step) {
            cameraLocked = false;
        }
    }

    glTranslatef(cam.translateX, cam.translateY, cam.translateZ);
    glRotatef(cam.rotateX, 1.0, 0.0, 0.0);
    glRotatef(cam.rotateY, 0.0, 1.0, 0.0);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display() {
    // run CUDA kernel to generate vertex positions
    if (animate) {
        runCuda();
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    adjustCamera();

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, posVertexBuffer);
    glVertexPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(1, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
    glClientActiveTexture(GL_TEXTURE1);
    glTexCoordPointer(2, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glUseProgram(shaderProg);

    // Set default uniform variables parameters for the vertex shader
    GLuint uniSize = glGetUniformLocation(shaderProg, "size");
    glUniform2f(uniSize, (float) meshSize, (float) meshSize);

    // Set default uniform variables parameters for the pixel shader
    GLuint uniWaterColor, uniLandColor, uniMountainColor, uniSkyColor, uniLightDir;

    uniWaterColor = glGetUniformLocation(shaderProg, "waterColor");
    glUniform4f(uniWaterColor, 0.0f, 0.1f, 0.4f, 1.0f);

    uniLandColor = glGetUniformLocation(shaderProg, "landColor");
    glUniform4f(uniLandColor, 0.0f, 0.4f, 0.0f, 1.0f);

    uniMountainColor = glGetUniformLocation(shaderProg, "mountainColor");
    glUniform4f(uniMountainColor, 0.8f, 0.8f, 0.8f, 1.0f);

    uniSkyColor = glGetUniformLocation(shaderProg, "skyColor");
    glUniform4f(uniSkyColor, 1.0f, 1.0f, 1.0f, 1.0f);

    uniLightDir = glGetUniformLocation(shaderProg, "lightDir");
    glUniform3f(uniLightDir, 0.0f, 1.0f, 0.0f);
    // end of uniform settings

    glColor3f(1.0, 1.0, 1.0);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLE_STRIP, ((meshSize * 2) + 2) * (meshSize - 1),
                   GL_UNSIGNED_INT, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


    glDisableClientState(GL_VERTEX_ARRAY);
    glClientActiveTexture(GL_TEXTURE0);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE1);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glUseProgram(0);

    glutSwapBuffers();
}

void timerEvent(int value) {
    float time = sdkGetTimerValue(&timer);

    if (animate) {
        animTime += (time - prevTime) * animationRate;
    }

    glutPostRedisplay();
    prevTime = time;

    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void cleanup() {
    sdkDeleteTimer(&timer);
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_heightVB_resource));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_slopeVB_resource));

    deleteVBO(&posVertexBuffer);
    deleteVBO(&heightVertexBuffer);
    deleteVBO(&slopeVertexBuffer);

    checkCudaErrors(cudaFree(d_slope));
    checkCudaErrors(cudaFree(d_heightMap));
    checkCudaErrors(cudaFree(d_heightMapNext));
    checkCudaErrors(cudaFreeHost(g_height));
}

void enterFollowMode() {
    followMode = true;
    cameraLocked = true;
    animate = true;
    oldCam = cam;
    followTriggeredAtStep = step;
}

void exitFollowMode() {
    followMode = false;
    followTriggeredAtStep = step;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
        case (27):
            cleanup();
            exit(EXIT_SUCCESS);
        case ' ':
            animate = !animate;
            break;
        case 'f':
            followMode ? exitFollowMode() : enterFollowMode();
            break;
        default:
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        mouseButtons |= 1 << button;
    } else if (state == GLUT_UP) {
        mouseButtons = 0;
    }

    mouseOldX = x;
    mouseOldY = y;
    glutPostRedisplay();
}

void motion(int x, int y) {
    if (cameraLocked) {
        mouseOldX = x;
        mouseOldY = y;
        return;
    }

    float dx, dy;
    dx = (float) (x - mouseOldX);
    dy = (float) (y - mouseOldY);

    if (mouseButtons == 1) {
        cam.rotateX += dy * 0.2f;
        cam.rotateY += dx * 0.2f;
    } else if (mouseButtons == 2) {
        cam.translateX += dx * 0.01f;
        cam.translateY -= dy * 0.01f;
    } else if (mouseButtons == 4) {
        cam.translateZ += dy * 0.01f;
    }

    mouseOldX = x;
    mouseOldY = y;
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(75.0, (double) w / (double) h, 0.1, 10.0);

    windowW = w;
    windowH = h;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv) {
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(windowW, windowH);
    glutCreateWindow("CUDA Random Terrain Generator");

    vertShaderPath = sdkFindFilePath("terrain.vert", argv[0]);
    fragShaderPath = sdkFindFilePath("terrain.frag", argv[0]);

    if (vertShaderPath == NULL || fragShaderPath == NULL) {
        fprintf(stderr, "Error unable to find GLSL vertex and fragment shaders!\n");
        exit(EXIT_FAILURE);
    }

    // initialize necessary OpenGL extensions

    if (!isGLVersionSupported(2, 0)) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    if (!areGLExtensionsSupported(
            "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Error: failed to get minimal extensions for terrain generator\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 2.0\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // load shader
    shaderProg = loadGLSLProgram(vertShaderPath, fragShaderPath);

    SDK_CHECK_ERROR_GL();
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, int size) {
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo) {
    glDeleteBuffers(1, vbo);
    *vbo = 0;
}

// create index buffer for rendering quad mesh
void createMeshIndexBuffer(GLuint *id, int w, int h) {
    int size = ((w * 2) + 2) * (h - 1) * sizeof(GLuint);

    // create index buffer
    glGenBuffers(1, id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    // fill with indices for rendering mesh as triangle strips
    auto *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

    if (!indices) {
        return;
    }

    for (int y = 0; y < h - 1; y++) {
        for (int x = 0; x < w; x++) {
            *indices++ = y * w + x;
            *indices++ = (y + 1) * w + x;
        }

        // start new strip with degenerate triangle
        *indices++ = (y + 1) * w + (w - 1);
        *indices++ = (y + 1) * w;
    }

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices
void createMeshPositionVBO(GLuint *id, int w, int h) {
    createVBO(id, w * h * 4 * sizeof(float));

    glBindBuffer(GL_ARRAY_BUFFER, *id);
    auto *pos = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    if (!pos) {
        return;
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float u = x / (float) (w - 1);
            float v = y / (float) (h - 1);
            *pos++ = u * 2.0f - 1.0f;
            *pos++ = 0.0f;
            *pos++ = v * 2.0f - 1.0f;
            *pos++ = 1.0f;
        }
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Attach shader to a program
int attachShader(GLuint prg, GLenum type, const char *name) {
    GLuint shader;
    FILE *fp;
    int size, compiled;
    char *src;

    fp = fopen(name, "rb");

    if (!fp) {
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    src = (char *) malloc(size);

    fseek(fp, 0, SEEK_SET);
    auto size_read = fread(src, sizeof(char), size, fp);
    fclose(fp);
    if (size_read != size) {
        fprintf(stderr, "Error reading shader file: %s\n", name);
        free(src);
        return 0;
    }

    shader = glCreateShader(type);
    glShaderSource(shader, 1, (const char **) &src, (const GLint *) &size);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint *) &compiled);

    if (!compiled) {
        char log[2048];
        int len;

        glGetShaderInfoLog(shader, 2048, (GLsizei *) &len, log);
        printf("Info log: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }

    free(src);

    glAttachShader(prg, shader);
    glDeleteShader(shader);

    return 1;
}

// Create shader program from vertex shader and fragment shader files
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName) {
    GLint linked;
    GLuint program;

    program = glCreateProgram();

    if (!attachShader(program, GL_VERTEX_SHADER, vertFileName)) {
        glDeleteProgram(program);
        fprintf(stderr, "Couldn't attach vertex shader from file %s\n",
                vertFileName);
        return 0;
    }

    if (!attachShader(program, GL_FRAGMENT_SHADER, fragFileName)) {
        glDeleteProgram(program);
        fprintf(stderr, "Couldn't attach fragment shader from file %s\n",
                fragFileName);
        return 0;
    }

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked) {
        glDeleteProgram(program);
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        fprintf(stderr, "Failed to link program: %s\n", temp);
        return 0;
    }

    return program;
}