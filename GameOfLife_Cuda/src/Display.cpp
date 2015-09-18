#include "Display.h"

#pragma comment(lib, "glew32s.lib")

#ifndef NDEBUG
#pragma comment(lib, "freeglutd.lib")
#else
#pragma comment(lib, "freeglut.lib")
#endif


//int GLDisplay::screen_x_= 1024;
//int GLDisplay::screen_y_ = 800;
int GLDisplay::win_width_ = 0;
int GLDisplay::win_height_ = 0;
GLuint GLDisplay::imageTex_ = 0u;
GLuint GLDisplay::imageBuffer_ = 0u;
struct cudaGraphicsResource* GLDisplay::imageBuffer_CUDA_;
float4* GLDisplay::grid_pixels_;
float4* GLDisplay::pixels_;
unsigned char GLDisplay::color_true_ = 125;
unsigned char GLDisplay::color_false_ = 0;
unsigned int GLDisplay::grid_width_ = 8 * 8 * 8; //2^10
unsigned int GLDisplay::grid_height_ = 8 * 8 * 8; //2^10
int GLDisplay::fill_thresold_ = 30;
Grid GLDisplay::cpu_grid_shared_ = Grid{};
Grid GLDisplay::grid_const_ = Grid{};
Grid GLDisplay::grid_computed_ = Grid{};
dim3 GLDisplay::grid_size_ = dim3();
dim3 GLDisplay::block_size_ = dim3();