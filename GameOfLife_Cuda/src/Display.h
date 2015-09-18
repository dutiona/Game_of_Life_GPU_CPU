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
    static void init(int* argc, char* argv[])
    {
		{//OpenGL init
			glutInit(argc, argv);
			//glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
			glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
			glutInitWindowPosition(0, 0);
			glutInitWindowSize(grid_width_, grid_height_);
			glutCreateWindow("Game of Life");
			glClearColor(120.0, 120.0, 120.0, 1.0);
			//glEnable(GL_DEPTH_TEST);
			glDisable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			// View Ortho
			// Sets up the OpenGL window so that (0,0) corresponds to the top left corner, 
			// and (SCREEN_X,SCREEN_Y) corresponds to the bottom right hand corner.  
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, 1024, 800, 0, 0, 1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization

			// Registering GL buffer for CUDA.
			GLint GlewInitResult = glewInit();
			if (GlewInitResult != GLEW_OK) {
				printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
			}
		}

		{//Register callback

			glutDisplayFunc(&GLDisplay::display);
			glutIdleFunc(&GLDisplay::idle);
			//glutMotionFunc(mouseMotion);
			//glutMouseFunc(mouse);
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
					const auto v = uniform_distrib(rd_mt_engine) < fill_thresold_;
					cpu_grid_shared_.grid[i*cpu_grid_shared_.width + j] = v;
				}
			}
			CudaSafeCall(
				cudaMemcpy(grid_const_.grid, cpu_grid_shared_.grid,
				grid_const_.width*grid_const_.height*sizeof(bool),
				cudaMemcpyHostToDevice));
			grid_size_ = dim3(grid_width_ / 8, grid_height_ / 8);
			block_size_ = dim3(8, 8);
			freeGrid(cpu_grid_shared_); //Plus besoin, openGL ira directement chercher la grille sur GPU

			CudaSafeCall(cudaMalloc(&grid_pixels_, grid_width_*grid_height_*sizeof(float4)));
			//pixels_ = static_cast<float4*>(malloc(grid_width_*grid_height_*sizeof(float4)));
		}
		

		{///Buffer init
			glGenBuffers(1, &imageBuffer_);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuffer_);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, grid_width_ * grid_height_ * sizeof(float4), 0, GL_DYNAMIC_COPY);
			CudaSafeCall(cudaGraphicsGLRegisterBuffer(&imageBuffer_CUDA_, imageBuffer_, cudaGraphicsRegisterFlagsWriteDiscard));
			
			glEnable(GL_TEXTURE_2D); // Enable Texturing
			glGenTextures(1, &imageTex_); // Generate a texture ID
			glBindTexture(GL_TEXTURE_2D, imageTex_); // Make this the current texture (GL is state-based)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, grid_width_, grid_height_, 0, GL_RGBA, GL_FLOAT, NULL); // Allocate the texture memory. The last parameter is NULL since we only want to allocate memory, not initialize it 
			// Must set the filter mode, GL_LINEAR enables interpolation when scaling 
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
    };

    static void run()
    {
        glutMainLoop();

		//Clean

		CudaSafeCall(cudaGraphicsUnregisterResource(imageBuffer_CUDA_));
		freeGridCuda(grid_const_);
		freeGridCuda(grid_computed_);

		glDeleteTextures(1, &imageTex_);
		glDeleteBuffers(1, &imageBuffer_);
    }

private:
	//static int screen_x_;
	//static int screen_y_;
	static int win_width_;
	static int win_height_;
	static GLuint imageTex_;
	static GLuint imageBuffer_;
	static struct cudaGraphicsResource* imageBuffer_CUDA_;
	static float4* grid_pixels_;
	static float4* pixels_;
	static unsigned char color_true_;
	static unsigned char color_false_;
	static unsigned int grid_width_;
	static unsigned int grid_height_;
	static int fill_thresold_;
	static Grid cpu_grid_shared_;
	static Grid grid_const_;
	static Grid grid_computed_;
	static dim3 grid_size_;
	static dim3 block_size_;

	static void idle(){
		glutPostRedisplay();
	}

	static void display(){

		
		// http://www.scribd.com/doc/84859529/57/OpenGL-Interoperability p.49
		// http://on-demand.gputechconf.com/gtc/2012/presentations/SS101B-Mixing-Graphics-Compute.pdf

		CudaSafeCall(cudaGraphicsMapResources(1, &imageBuffer_CUDA_, 0));
		size_t num_bytes;
		CudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&grid_pixels_, &num_bytes, imageBuffer_CUDA_));
		

		float4 color_true; color_true.x = color_true_; color_true.y = color_true_; color_true.z = color_true_; color_true.w = 1.0;
		float4 color_false; color_false.x = color_false_; color_false.y = color_false_; color_false.z = color_false_; color_false.w = 1.0;

		//Kernel call
		do_step_shared_gl(grid_size_, block_size_, grid_const_, grid_computed_, grid_pixels_, color_true, color_false);
		//CudaSafeCall(cudaMemcpy(pixels_, grid_pixels_, grid_width_*grid_height_*sizeof(float4), cudaMemcpyDeviceToHost));


		//juliaKernel << <dimGrid, dimBlock >> >(cupixels, cuseedr, cuseedi, cuprecision, cuscale);

		CudaSafeCall(cudaGraphicsUnmapResources(1, &imageBuffer_CUDA_));

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// http://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuffer_); // Select the appropriate buffer 	
		glBindTexture(GL_TEXTURE_2D, imageTex_); // Select the appropriate texture	
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, grid_width_, grid_height_, GL_RGBA, GL_FLOAT, NULL); // Make a texture from the buffer

		glBegin(GL_QUADS);
			glTexCoord2f(0, 1.0f);
			glVertex3f(0, 0, 0);
			glTexCoord2f(0, 0);
			glVertex3f(0, grid_height_, 0);
			glTexCoord2f(1.0f, 0);
			glVertex3f(grid_width_, grid_height_, 0);
			glTexCoord2f(1.0f, 1.0f);
			glVertex3f(grid_width_, 0, 0);
		glEnd();
		


		/*

		float4 color_true; color_true.x = color_true_; color_true.y = color_true_; color_true.z = color_true_; color_true.w = 1.0;
		float4 color_false; color_false.x = color_false_; color_false.y = color_false_; color_false.z = color_false_; color_false.w = 1.0;

		//Kernel call
		do_step_shared_gl(grid_size_, block_size_, grid_const_, grid_computed_, grid_pixels_, color_true, color_false);
		CudaSafeCall(cudaMemcpy(pixels_, grid_pixels_, grid_width_*grid_height_*sizeof(float4), cudaMemcpyDeviceToHost));
		//glDrawPixels(grid_width_, grid_height_, GL_RGBA, GL_FLOAT, pixels_);

		glutSwapBuffers();

		*/

		//Fin boucle affichage





		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//
		//glMatrixMode(GL_MODELVIEW);
		//glLoadIdentity();
		//glTranslatef(0.f, 0.f, -1.f); /* eye position */
		//
		//glEnable(GL_TEXTURE_2D); /* enable texture mapping */
		//glBindTexture(GL_TEXTURE_2D, texture_); /* bind to our texture, has id of 13 */
		//
		//glBegin(GL_QUADS);
		//glTexCoord2f(0.0f, 0.0f); /* lower left corner of image */
		//glVertex3f(-1.f, -1.f, 0.0f);
		//glTexCoord2f(1.0f, 0.0f); /* lower right corner of image */
		//glVertex3f(1.f, -1.f, 0.0f);
		//glTexCoord2f(1.0f, 1.0f); /* upper right corner of image */
		//glVertex3f(1.f, 1.f, 0.0f);
		//glTexCoord2f(0.0f, 1.0f); /* upper left corner of image */
		//glVertex3f(-1.f, 1.0f, 0.0f);
		//glEnd();
		//
		//glDisable(GL_TEXTURE_2D); /* disable texture mapping */
		//glutSwapBuffers();
	}
};
