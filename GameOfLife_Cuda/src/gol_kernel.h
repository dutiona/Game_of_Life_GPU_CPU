#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

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
	size_t width;
	size_t height;
	bool* grid;
} Grid;

__host__ void initGrid(Grid& g, size_t w, size_t h);
__host__ void initGridCuda(Grid& g, size_t w, size_t h);
__host__ void freeGrid(Grid& g);
__host__ void freeGridCuda(Grid& g);

__host__ void printGrid(Grid& grid);

__host__ const Grid* do_step(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed);

__device__ inline size_t countAliveNeighbours(size_t x, size_t y, const Grid& g);

__global__ void gol_step_kernel(const Grid grid_const, Grid grid_computed);