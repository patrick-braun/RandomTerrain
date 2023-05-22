cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-Wl,--copy-dt-needed-entries")

project(RandomTerrain LANGUAGES CUDA CXX)
enable_language(CUDA)


add_executable(RandomTerrain RandomTerrain.cpp RandomTerrain_kernel.cu)
set_target_properties(RandomTerrain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(RandomTerrain PRIVATE include)
target_include_directories(RandomTerrain PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

target_link_libraries(RandomTerrain PUBLIC OpenGL::OpenGL OpenGL::GLU GLUT::GLUT GLEW::GLEW)