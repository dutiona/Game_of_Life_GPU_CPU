
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void gol_step_kernel(bool *grid_start, bool* grid_tmp/*, size_t width, size_t height*/)
{
    size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	size_t width = blockDim.x * gridDim.x;
	size_t height = blockDim.y * gridDim.y;

	//récupérer les voisins sur grid start

	//Mettre à jour létat en x,y sur grid_tmp

    //c[i] = a[i] + b[i];
}

int main()
{
	size_t nb_loop = 100;
	size_t width, height;

	bool* _cpu_pointer,

	bool* grid_const;
	bool* grid_computed;
	CudaSafeCall(cudaMalloc(&grid_const, width*height*sizeof(bool)));
	CudaSafeCall(cudaMalloc(&grid_computed, width*height*sizeof(bool)));

	CudaSafeCall(cudaMemcpy(grid_const, _cpu_pointer, width*height*sizeof(bool), cudaMemcpyHostToDevice));

	dim3 grid_size = dim3(width/8, height/8);
	dim3 block_size = dim3(8, 8);

	for (int i = 0; i < nb_loop; ++i){
		gol_step_kernel <<< grid_size, block_size >>> (grid_const, grid_computed);
		auto tmp = grid_computed;
		grid_computed = grid_const;
		grid_const = tmp;
	}


	CudaSafeCall(cudaMemcpy(_cpu_pointer, grid_const, width*height*sizeof(bool), cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaFree(grid_const));
	CudaSafeCall(cudaFree(grid_computed));

    return 0;
}
