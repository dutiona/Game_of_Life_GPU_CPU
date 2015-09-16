#include "Display.h"

#pragma comment(lib, "glew32s.lib")

#ifndef NDEBUG
#pragma comment(lib, "freeglutd.lib")
#else
#pragma comment(lib, "freeglut.lib")
#endif

int GLDisplay::win_width = 0;
int GLDisplay::win_height = 0;
GLuint GLDisplay::texture = 0u;
GLuint GLDisplay::buffer = 0u;