#pragma once

#include <vector>

#define GLEW_STATIC

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void initTexture(int imageWidth, int imageHeight, unsigned char *h_data);

class GLDisplay
{
public:
    static void init(const float * data, const int x_size, const int y_size, const int x_output_size, const int y_output_size)
    {
        float * local_data = (float*)malloc(x_size * y_size * sizeof(float));

        float max = 0.f;

        for(int i = 0; i < x_size; i++)
        {
            for(int j = 0; j < y_size; j++)
            {
                int index = (x_size - 1 - i) * y_size + j;

                local_data[index] = data[i * y_size + j];
                if(local_data[index] > max)
                    max = local_data[index];
            }
        }

        for(int i = 0; i < x_size; i++)
        {
            for(int j = 0; j < y_size; j++)
            {
                int index = (x_size - 1 - i) * y_size + j;
                local_data[index] /= max;
            }
        }

        void * dev_input_data_ptr;
        void * dev_tmp_ptr;
        cudaMalloc(&dev_input_data_ptr, x_size * y_size * sizeof(float));
        cudaMalloc(&dev_tmp_ptr, x_output_size * y_output_size * sizeof(float));
        cudaMemcpy(dev_input_data_ptr, local_data, x_size * y_size * sizeof(float), cudaMemcpyHostToDevice);

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
        glBufferData(GL_PIXEL_UNPACK_BUFFER, x_output_size * y_output_size * 3, NULL, GL_DYNAMIC_COPY);
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, x_output_size, y_output_size, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);


        void * dev_output_data_ptr;
        cudaGLMapBufferObject(&dev_output_data_ptr, buffer);

        //compute_image((float*)dev_input_data_ptr, x_size, y_size, (float*)dev_tmp_ptr, (unsigned char*)dev_output_data_ptr, x_output_size, y_output_size);

        cudaGLUnmapBufferObject(buffer);


        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, x_output_size, y_output_size, GL_RGB, GL_UNSIGNED_BYTE, NULL);
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
