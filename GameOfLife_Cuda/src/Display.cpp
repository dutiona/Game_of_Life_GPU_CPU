#include "Display.h"

#pragma comment(lib, "glew32s.lib")

#ifndef NDEBUG
#pragma comment(lib, "freeglutd.lib")
#else
#pragma comment(lib, "freeglut.lib")
#endif

//Variables privées
Grid GLDisplay::cpu_grid_shared_ = Grid{};
Grid GLDisplay::grid_const_ = Grid{};
Grid GLDisplay::grid_computed_ = Grid{};
uchar4* GLDisplay::grid_pixels_;
uchar4* GLDisplay::pixels_;
dim3 GLDisplay::grid_size_ = dim3();
dim3 GLDisplay::block_size_ = dim3();
int GLDisplay::frame_ = 0;
int GLDisplay::timebase_ = 0;
GLuint GLDisplay::imageTex_ = 0u;
GLuint GLDisplay::imageBuffer_ = 0u;


//Configuration
int GLDisplay::screen_x_ = 1024; //Taille de l'écran
int GLDisplay::screen_y_ = 800;
bool GLDisplay::interop_ = true; //Reshape automatique non géré en mode non interop
uchar4 GLDisplay::color_true_ = uchar4{ 255, 0, 0, 255 }; //Couleurs des pixels
uchar4 GLDisplay::color_false_ = uchar4{ 125, 125, 125, 255 };
unsigned int GLDisplay::grid_width_ = 8 * 8 * 4; //2^8
unsigned int GLDisplay::grid_height_ = 8 * 8 * 4; //2^8
int GLDisplay::fill_thresold_ = 70; //% de remplissage
