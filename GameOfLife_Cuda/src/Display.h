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
		//Cuda init
		size_t nb_loop = 10000;
		unsigned int width = 2 * 2 * 2 * 2 * 2 * 2 * 2; //2^7
		unsigned int height = 2 * 2 * 2 * 2 * 2 * 2 * 2; //2^7
		int fill_thresold = 30;

		Grid cpu_grid;
		initGrid(cpu_grid, width, height);

		//Random init
		const auto seed = std::random_device{}(); //seed ne dépend pas de std::chrono
		std::mt19937 rd_mt_engine(seed); // mt19937 est le mersenne_twister_engine standard
		std::uniform_int_distribution<int> uniform_distrib(1, 100); // distribution 1 à 100 uniforme

		for (unsigned int i = 0; i < cpu_grid.width; ++i){
			for (unsigned int j = 0; j < cpu_grid.height; ++j){
				//Remplissage aléatoire de la grille en fonction du fill_thresold
				cpu_grid.grid[i*cpu_grid.width + j] = uniform_distrib(rd_mt_engine) < fill_thresold;
			}
		}

		Grid grid_const;
		Grid grid_computed;
		initGridCuda(grid_const, width, height);
		initGridCuda(grid_computed, width, height);

		CudaSafeCall(
			cudaMemcpy(grid_const.grid, cpu_grid.grid,
			grid_const.width*grid_const.height*sizeof(bool),
			cudaMemcpyHostToDevice));

		dim3 grid_size = dim3(width / 8, height / 8);
		dim3 block_size = dim3(8, 8);

		//OpenGL init
		glutInit(argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
        glutInitWindowSize(win_width, win_height);
        glutCreateWindow("Résultats de champ");
        glutDisplayFunc(&GLDisplay::display);
        glutReshapeFunc(&GLDisplay::reshape);
        glEnable(GL_DEPTH_TEST);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Registering GL buffer for CUDA.
        glewInit();
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, cpu_grid.width * cpu_grid.height * 3, NULL, GL_DYNAMIC_COPY);
        cudaGLRegisterBufferObject(buffer);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cpu_grid.width, cpu_grid.height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);


        void* color_buffer;
		cudaGLMapBufferObject(&color_buffer, buffer);

		//Compute pixels
		for (size_t i = 0; i < nb_loop; ++i){
			do_step_gl(grid_size, block_size, grid_const, grid_computed, (unsigned char*)color_buffer, 255, 0);
		}

		CudaSafeCall(
			cudaMemcpy(cpu_grid.grid, grid_const.grid,
			cpu_grid.width*cpu_grid.height*sizeof(bool),
			cudaMemcpyDeviceToHost));

		freeGrid(cpu_grid);
		freeGridCuda(grid_const);
		freeGridCuda(grid_computed);


        cudaGLUnmapBufferObject(buffer);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glBindTexture(GL_TEXTURE_2D, texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cpu_grid.width, cpu_grid.height, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    };

    static void run()
    {
        glutMainLoop();
    }

private:
	static int win_width;
	static int win_height;
	static GLuint texture;
	static GLuint buffer;

	static void reshape(int w, int h)
	{
		win_width = (w + h) / 2;
		win_height = (h + w) / 2;

		glViewport(0, 0, win_width, win_height);

		glutReshapeWindow(win_width, win_height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		gluPerspective(90, win_width / win_height, 1, 9999);

		glutPostRedisplay();
	}

	static void display(void)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.f, 0.f, -1.f); /* eye position */

		glEnable(GL_TEXTURE_2D); /* enable texture mapping */
		glBindTexture(GL_TEXTURE_2D, texture); /* bind to our texture, has id of 13 */

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
		glutSwapBuffers();
	}
};
