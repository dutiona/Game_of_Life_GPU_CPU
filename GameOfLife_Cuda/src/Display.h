#pragma once

#include <vector>
#include <random>

#define GLEW_STATIC

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "gol_kernel.cuh"

class GLDisplay
{
public:
	static void init(int* argc, char* argv[]){
		{//OpenGL init
			glutInit(argc, argv);
			glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
			glutInitWindowPosition(0, 0);
			glutInitWindowSize(grid_width_, grid_height_);
			glutCreateWindow("Game of Life");
			glClearColor(120.0, 120.0, 120.0, 1.0);
			glDisable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			// Registering GL buffer for CUDA.
			GLint GlewInitResult = glewInit();
			if (GlewInitResult != GLEW_OK) {
				printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
			}
		}

		{//Register callback
			glutDisplayFunc(&GLDisplay::display);
			glutIdleFunc(&GLDisplay::idle);
			//glutKeyboardFunc(processNormalKeys);
			//glutSpecialFunc(processSpecialKeys);
		}

		{///Init cuda grid random
			initGrid(cpu_grid_shared_, grid_width_, grid_height_);
			initGridCuda(grid_const_, grid_width_, grid_height_);
			initGridCuda(grid_computed_, grid_width_, grid_height_);
			const auto seed = std::random_device{}(); //seed ne dépend pas de std::chrono
			auto rd_mt_engine = std::mt19937{ seed }; // mt19937 est le mersenne_twister_engine standard
			auto uniform_distrib = std::uniform_int_distribution<int>{1, 100}; // distribution 1 à 100 uniforme
			for (unsigned int i = 0; i < grid_width_; ++i){
				for (unsigned int j = 0; j < grid_width_; ++j){
					//Remplissage aléatoire de la grille en fonction du fill_thresold
					cpu_grid_shared_.grid[i*cpu_grid_shared_.width + j] = uniform_distrib(rd_mt_engine) < fill_thresold_;
				}
			}
			CudaSafeCall(
				cudaMemcpy(grid_const_.grid, cpu_grid_shared_.grid,
				grid_const_.width*grid_const_.height*sizeof(bool),
				cudaMemcpyHostToDevice));
			grid_size_ = dim3(grid_width_ / 8, grid_height_ / 8);
			block_size_ = dim3(8, 8);
			freeGrid(cpu_grid_shared_); //Plus besoin, openGL ira directement chercher la grille sur GPU

			if (!interop_){
				CudaSafeCall(cudaMalloc(&grid_pixels_, grid_width_*grid_height_*sizeof(uchar4)));
				pixels_ = static_cast<uchar4*>(malloc(grid_width_*grid_height_*sizeof(uchar4)));
			}
		}

		if (interop_){
			///Buffer init
			glGenBuffers(1, &imageBuffer_);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuffer_);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, grid_width_ * grid_height_ * sizeof(uchar4), NULL, GL_DYNAMIC_COPY);

			//Allocation mémoire GPU
			cudaGLRegisterBufferObject(imageBuffer_);

			glGenTextures(1, &imageTex_);
			glBindTexture(GL_TEXTURE_2D, imageTex_);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, grid_width_, grid_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		}
	};

	static void run()
	{
		glutMainLoop();

		//Clean
		freeGridCuda(grid_const_);
		freeGridCuda(grid_computed_);

		if (interop_){
			glDeleteTextures(1, &imageTex_);
			glDeleteBuffers(1, &imageBuffer_);
		}
		else{
			free(pixels_);
		}
	}

private:
	static bool interop_;
	static int win_width_;
	static int win_height_;
	static GLuint imageTex_;
	static GLuint imageBuffer_;
	static struct cudaGraphicsResource* imageBuffer_CUDA_;
	static uchar4* grid_pixels_;
	static uchar4* pixels_;
	static uchar4 color_true_;
	static uchar4 color_false_;
	static unsigned int grid_width_;
	static unsigned int grid_height_;
	static int fill_thresold_;
	static Grid cpu_grid_shared_;
	static Grid grid_const_;
	static Grid grid_computed_;
	static dim3 grid_size_;
	static dim3 block_size_;
	static int frame_;
	static int timebase_;

	static void idle(){
		glutPostRedisplay();
	}

	static void display(){

		frame_++;
		int timecur = glutGet(GLUT_ELAPSED_TIME);

		if (timecur - timebase_ > 500) {
			char t[200];
			char* m = "";
			sprintf(t, "%s:  %s, %s mode, (%.2f) FPS", "Game of Life", m, interop_ ? "interop" : "gpu", frame_ * 1000 / (double)(timecur - timebase_));
			glutSetWindowTitle(t);
			timebase_ = timecur;
			frame_ = 0;
		}

		if (interop_){
			// Mapping de l'alloc GPU sur un pointeur à envoyer au kernel et verouillage du buffer par cuda
			cudaGLMapBufferObject((void**)&grid_pixels_, imageBuffer_);

			//Kernel call
			do_step_shared_gl(grid_size_, block_size_, grid_const_, grid_computed_, grid_pixels_, color_true_, color_false_);

			// Dissociation de cuda du buffer pour pouvoir autoriser openGL à lire dedans
			cudaGLUnmapBufferObject(imageBuffer_);

			//Bind du buffer par OpenGL
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuffer_);
			glBindTexture(GL_TEXTURE_2D, imageTex_);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, grid_width_, grid_height_, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

			//Affichage
			glClear(GL_COLOR_BUFFER_BIT);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(0.f, 0.f, -1.f); /* eye position */

			glEnable(GL_TEXTURE_2D); /* enable texture mapping */
			glBindTexture(GL_TEXTURE_2D, imageTex_); /* bind to our texture, has id of 13 */

			//Placage de la texture dans la primitive Quad
			glBegin(GL_QUADS);
				glTexCoord2f(0.0f, 0.0f); /* lower left corner of image */
				glVertex3f(-1.f, -1.f, 0.0f);
				glTexCoord2f(1.0f, 0.0f); /* lower right corner of image */
				glVertex3f(1.f, -1.f, 0.0f);
				glTexCoord2f(1.0f, 1.0f); /* upper right corner of image */
				glVertex3f(1.f, 1.f, 0.0f);
				glTexCoord2f(0.0f, 1.0f); /* upper left corner of image */
				glVertex3f(-1.f, 1.0f, 0.0f);
			glEnd();

			glDisable(GL_TEXTURE_2D); /* disable texture mapping */
		}
		else{
			do_step_shared_gl(grid_size_, block_size_, grid_const_, grid_computed_, grid_pixels_, color_true_, color_false_);
			CudaSafeCall(cudaMemcpy(pixels_, grid_pixels_, grid_width_*grid_height_*sizeof(uchar4), cudaMemcpyDeviceToHost));
			glDrawPixels(grid_width_, grid_height_, GL_RGBA, GL_UNSIGNED_BYTE, pixels_);
		}
		glutSwapBuffers();


	}
};
