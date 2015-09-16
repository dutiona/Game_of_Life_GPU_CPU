#include <stdio.h>

#define GLEW_STATIC

// OpenGL Graphics includes
#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#pragma comment(lib, "glew32s.lib")

#ifndef NDEBUG
#pragma comment(lib, "freeglutd.lib")
#else
#pragma comment(lib, "freeglut.lib")
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define SCREEN_X 1024
#define SCREEN_Y 768
#define FPS_UPDATE 500
#define TITLE "Julia"

#define CPU_MODE 1
#define GPU_MODE 2
#define INTEROP_MODE 3


struct complex {
	double r;
	double i;
	__host__ __device__ complex(double a, double b) : r(a), i(b) {}
	__host__ __device__ double magnitude2 (void) { return r * r + i * i;}
	__host__ __device__ complex operator*(const complex& a) { return complex(r*a.r - i*a.i, i*a.r + r*a.i); }
	__host__ __device__ complex operator+(const complex& a) { return complex(r+a.r, i+a.i); }
};

GLuint imageTex;
GLuint imageBuffer;
double* debug;
struct cudaGraphicsResource* imageBuffer_CUDA;

/* Globals */
complex seed = complex(-0.8, 0.156);
double scale = 0.003f;
int precision = 100;
int mode = GPU_MODE;
int frame=0;
int timebase=0;

double4 *pixels, *cupixels;
double *cuseedr, *cuseedi, *cuscale;
int *cuprecision;

__host__ __device__
void juliaColor(double4* pixel, double x, double y, double seedr, double seedi, int precision) {
	complex a(x, y);
	complex seed(seedr,seedi);
	pixel->z = 1.0f;
	pixel->w = 1.0f;
	int i;
	for (i=0; i<precision; i++) {
		a = a * a + seed;
		if (a.magnitude2() > 4) {
			double c = 1-i/(double)precision;
			pixel->x = c;
			pixel->y = c;
			return;
		}
	}
	pixel->x = 0.0f;
	pixel->y = 0.0f;
}

__global__
void juliaKernel(double4* pixel, double* seedr, double* seedi, int* precision, double* cuscale)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = j*SCREEN_X+i;
	if (index<SCREEN_X*SCREEN_Y) {
		double x = (double)(*cuscale*(i-SCREEN_X/2));
		double y = (double)(*cuscale*(j-SCREEN_Y/2));
		juliaColor(pixel+index,x,y,*seedr,*seedi,*precision);
	}
}

void initCPU()
{
	pixels = (double4*)malloc(SCREEN_X*SCREEN_Y*sizeof(double4));
}

void cleanCPU()
{
	free(pixels);
}


void initGPU()
{
	cudaMalloc((void **)&cuseedr, sizeof(double));
	cudaMalloc((void **)&cuseedi, sizeof(double));
	cudaMalloc((void **)&cuscale, sizeof(double));
	cudaMalloc((void **)&cuprecision, sizeof(int));
	cudaMalloc((void **)&cupixels, SCREEN_X*SCREEN_Y*sizeof(double4));
	pixels = (double4*)malloc(SCREEN_X*SCREEN_Y*sizeof(double4));
}

void cleanGPU()
{
	cudaFree(cuseedr); 
	cudaFree(cuseedi); 
	cudaFree(cuscale); 
	cudaFree(cuprecision); 
	cudaFree(cupixels); 
	free(pixels);
}

void initInterop()
{
	cudaMalloc((void **)&cuseedr, sizeof(double));
	cudaMalloc((void **)&cuseedi, sizeof(double));
	cudaMalloc((void **)&cuscale, sizeof(double));
	cudaMalloc((void **)&cuprecision, sizeof(int));

	cudaGLSetGLDevice(0); // Explicitly set device 0
	glGenBuffers(1, &imageBuffer); 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, SCREEN_X * SCREEN_Y * sizeof(double4), 0, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&imageBuffer_CUDA,imageBuffer,cudaGraphicsMapFlagsWriteDiscard);

	glEnable(GL_TEXTURE_2D); // Enable Texturing
	glGenTextures(1,&imageTex); // Generate a texture ID
	glBindTexture(GL_TEXTURE_2D, imageTex); // Make this the current texture (GL is state-based)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCREEN_X, SCREEN_Y, 0, GL_RGBA, GL_FLOAT, NULL); // Allocate the texture memory. The last parameter is NULL since we only want to allocate memory, not initialize it 
	// Must set the filter mode, GL_LINEAR enables interpolation when scaling 
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
}

void cleanInterop(){

	cudaGraphicsUnregisterResource(imageBuffer_CUDA);
	cudaFree(cuseedr); 
	cudaFree(cuseedi); 
	cudaFree(cuscale); 
	cudaFree(cuprecision); 

	glDeleteTextures(1, &imageTex);
	glDeleteBuffers(1, &imageBuffer);
}

void juliaInterop()
{
	// http://www.scribd.com/doc/84859529/57/OpenGL-Interoperability p.49
	// http://on-demand.gputechconf.com/gtc/2012/presentations/SS101B-Mixing-Graphics-Compute.pdf

	cudaMemcpy(cuseedr, &seed.r, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuseedi, &seed.i, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuscale, &scale, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuprecision, &precision, sizeof(int), cudaMemcpyHostToDevice);
	
	cudaGraphicsMapResources(1, &imageBuffer_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&cupixels,&num_bytes,imageBuffer_CUDA);

	// Execute kernel
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(SCREEN_X/dimBlock.x, SCREEN_Y/dimBlock.y, 1);
	juliaKernel<<<dimGrid, dimBlock>>>(cupixels, cuseedr, cuseedi,  cuprecision, cuscale);
		
	cudaGraphicsUnmapResources(1, &imageBuffer_CUDA);
}

void juliaGPU()
{
	cudaMemcpy(cuseedr, &seed.r, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuseedi, &seed.i, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuscale, &scale, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuprecision, &precision, sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(16, 16); // 256 threads
	dim3 numBlocks((SCREEN_X+threadsPerBlock.x-1)/threadsPerBlock.x,(SCREEN_Y+threadsPerBlock.y-1)/threadsPerBlock.y); 

	juliaKernel<<<numBlocks,threadsPerBlock>>>(cupixels, cuseedr, cuseedi,  cuprecision, cuscale);

	cudaMemcpy(pixels, cupixels, SCREEN_X*SCREEN_Y*sizeof(double4), cudaMemcpyDeviceToHost);	
}

void juliaCPU()
{
	int i,j;
	for (i=0;i<SCREEN_Y;i++)
		for (j=0;j<SCREEN_X;j++)
		{
			double x = (double)(scale*(j-SCREEN_X/2));
			double y = (double)(scale*(i-SCREEN_Y/2));
			juliaColor(pixels+(i*SCREEN_X+j),x,y,seed.r,seed.i,precision);
		}
}

void calcJulia() {
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m="";
		switch (mode)
		{
			case CPU_MODE: m = "CPU"; break;
			case GPU_MODE: m = "GPU"; break;
			case INTEROP_MODE: m = "INTEROP"; break;
		}
		sprintf(t,"%s:  %s, %i loops, %.2f FPS",TITLE,m,precision,frame*1000/(double)(timecur-timebase));
		glutSetWindowTitle(t);
	 	timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
		case CPU_MODE: juliaCPU(); break;
		case GPU_MODE: juliaGPU(); break;
		case INTEROP_MODE: juliaInterop(); break;
	}
}

void idleJulia()
{
	glutPostRedisplay();
}


void renderJulia()
{	
	calcJulia();
	if (mode==INTEROP_MODE)
	{
		// http://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuffer); // Select the appropriate buffer 	
		glBindTexture(GL_TEXTURE_2D, imageTex); // Select the appropriate texture	
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,  SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, NULL); // Make a texture from the buffer
		glBegin(GL_QUADS);
			glTexCoord2f( 0, 1.0f); 
			glVertex3f(0,0,0);
			glTexCoord2f(0,0);
			glVertex3f(0,SCREEN_Y,0);
			glTexCoord2f(1.0f,0);
			glVertex3f(SCREEN_X,SCREEN_Y,0);
			glTexCoord2f(1.0f,1.0f);
			glVertex3f(SCREEN_X,0,0);
		glEnd();
	}
	else
	{
		glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels);
	}
	glutSwapBuffers();
}

void clean()
{
	switch (mode)
	{
		case CPU_MODE: cleanCPU(); break;
		case GPU_MODE: cleanGPU(); break;
		case INTEROP_MODE: cleanInterop(); break;
	}
}

void init()
{
	switch (mode)
	{
		case CPU_MODE: initCPU(); break;
		case GPU_MODE: initGPU(); break;
		case INTEROP_MODE: initInterop(); break;
	}

}

void toggleMode(int m)
{
	clean();
	mode = m;
	init();
}

void mouse(int button, int state, int x, int y)
{
	if (button<=2) 
	{
		seed.r = (double)(scale*(x-SCREEN_X/2));
		seed.i = -(double)(scale*(y-SCREEN_Y/2));
	}
	// Wheel reports as button 3 (scroll up) and button 4 (scroll down)
	if (button == 3) scale /= 1.05f;
	else if (button == 4) scale *= 1.05f;
}

void mouseMotion(int x, int y)
{
	seed.r = (double)(scale*(x-SCREEN_X/2));
	seed.i = -(double)(scale*(y-SCREEN_Y/2));
}

void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27) exit(0);
	else if (key=='1') toggleMode(CPU_MODE);
	else if (key=='2') toggleMode(GPU_MODE);
	else if (key=='3') toggleMode(INTEROP_MODE);
}

void processSpecialKeys(int key, int x, int y) {
	switch(key) {
		case GLUT_KEY_UP: if (precision <10) precision ++; else precision += 10; break;
		case GLUT_KEY_DOWN: if (precision>10) precision -= 10; else if (precision>1) precision--; break;
	}
}

void initGL(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0,0);
	glutInitWindowSize(SCREEN_X,SCREEN_Y);
	glutCreateWindow(TITLE);
	glClearColor(0.0,0.0,0.0,0.0);
	glDisable(GL_DEPTH_TEST);

	// View Ortho
	// Sets up the OpenGL window so that (0,0) corresponds to the top left corner, 
	// and (SCREEN_X,SCREEN_Y) corresponds to the bottom right hand corner.  
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	glOrtho (0, SCREEN_X, SCREEN_Y, 0, 0, 1);
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization
}


int main(int argc, char **argv) {

	initGL(argc, argv);

	init();
	toggleMode(GPU_MODE);

	glutDisplayFunc(renderJulia);
	glutIdleFunc(idleJulia);
  	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	GLint GlewInitResult = glewInit();
	if (GlewInitResult != GLEW_OK) {
		printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
	}

	// enter GLUT event processing cycle
	glutMainLoop();

	clean();

	return 1;
}
