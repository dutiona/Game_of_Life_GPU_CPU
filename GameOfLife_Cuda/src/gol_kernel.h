#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

// Define this to turn on error checking
#ifndef NDEBUG
#define CUDA_ERROR_CHECK
#endif

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line){
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

inline void __cudaCheckError(const char *file, const int line){
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

typedef struct{
	unsigned int width;
	unsigned int height;
	bool* grid;
} Grid;
__host__ void initGrid(Grid& g, unsigned int w, unsigned int h);
__host__ void initGridCuda(Grid& g, unsigned int w, unsigned int h);
__host__ void freeGrid(Grid& g);
__host__ void freeGridCuda(Grid& g);


//To OpenGL display
__host__ void do_step_gl(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed, unsigned char* colorBuffer, const char& color_true, const char& color_false);
__global__ void gol_step_kernel_gl(const Grid grid_const, Grid grid_computed, float* colorBuffer, const char& color_true, const char& color_false);


//To profile
__host__ void do_step(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed);
__global__ void gol_step_kernel(const Grid grid_const, Grid grid_computed);


//Internal
__device__ inline size_t countAliveNeighbours(unsigned int x, unsigned int y, const Grid& g);
__host__ void printGrid(Grid& grid);
__host__ void launch_kernel(const Grid& cpu_grid, size_t nb_loop, unsigned int width, unsigned int height);




