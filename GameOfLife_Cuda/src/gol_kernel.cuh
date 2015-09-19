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
	int width;
	int height;
	bool* grid;
} Grid;
__host__ void initGrid(Grid& g, unsigned int w, unsigned int h);
__host__ void initGridCuda(Grid& g, unsigned int w, unsigned int h);
__host__ void freeGrid(Grid& g);
__host__ void freeGridCuda(Grid& g);


//Global
__host__ void do_step(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed);
__global__ void gol_step_kernel(Grid grid_const, Grid grid_computed);
__host__ void launch_kernel(const Grid& cpu_grid, size_t nb_loop, unsigned int width, unsigned int height);

//Shared
__host__ void do_step_shared(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed);
__global__ void gol_step_kernel_shared(Grid grid_const, Grid grid_computed);
__host__ void launch_kernel_shared(const Grid& cpu_grid, size_t nb_loop, unsigned int width, unsigned int height);

//OpenGl
__host__ void do_step_shared_gl(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed, uchar4* grid_pixels, uchar4 color_true, uchar4 color_false);
__global__ void gol_step_kernel_shared_gl(const Grid grid_const, Grid grid_computed, uchar4* grid_pixels, uchar4 color_true, uchar4 color_false);

//Internal
__host__ void printGrid(Grid& grid);
__host__ bool gridAreEquals(const Grid& glhs, const Grid& grhs);

