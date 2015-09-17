#include "gol_kernel.h"

#include <Windows.h>
#include <time.h>

__host__ void initGrid(Grid& g, unsigned int w, unsigned int h){
	g.width = w;
	g.height = h;
	g.grid = (bool*)malloc(w*h*sizeof(bool));
}
__host__ void initGridCuda(Grid& g, unsigned int w, unsigned int h){
	g.width = w;
	g.height = h;
	CudaSafeCall(cudaMalloc(&g.grid, w*h*sizeof(bool)));
}
__host__ void freeGrid(Grid& g){
	free(g.grid);
}
__host__ void freeGridCuda(Grid& g){
	CudaSafeCall(cudaFree(g.grid));
}

__device__ inline size_t countAliveNeighbour(unsigned int x, unsigned int y, const Grid& g){
	unsigned int width = blockDim.x * gridDim.x;
	unsigned int height = blockDim.y * gridDim.y;
	int x_min_1 = ((x - 1) + width) % width;
	int y_min_1 = ((y - 1) + height) % height;
	int x_plus_1 = ((x + 1) + width) % width;
	int y_plus_1 = ((y + 1) + height) % height;
	return g.grid[x_min_1*g.width + y_min_1] + g.grid[x_min_1*g.width + y] + g.grid[x_min_1*g.width + y_plus_1] +
		g.grid[x*g.width + (y - 1)] + g.grid[x*g.width + (y + 1)] +
		g.grid[x_plus_1*g.width + y_min_1] + g.grid[x_plus_1*g.width + y] + g.grid[x_plus_1*g.width + y_plus_1];
}

__global__ void gol_step_kernel(const Grid grid_const, Grid grid_computed){
	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	size_t width = blockDim.x * gridDim.x;
	//size_t height = blockDim.y * gridDim.y;

	//récupérer les voisins sur grid start
	size_t cells_alive = countAliveNeighbours(x, y, grid_const);

	grid_computed.grid[x*width + y] =
		(grid_const.grid[x*width + y]) //alive
		? (
			(cells_alive < 2 || cells_alive > 3)
			? false //kill
			: grid_const.grid[x*width + y] //forward
		)
		: ( // dead
			(cells_alive == 3)
			? true //resurect
			: grid_const.grid[x*width + y] //forward
		);
}

__global__ void gol_step_kernel_shared(const Grid grid_const, Grid grid_computed){
	extern __shared__ bool grid_[];

	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	size_t width = blockDim.x * gridDim.x;
	size_t height = blockDim.y * gridDim.y;

	int x_min_1 = ((x - 1) + width) % width;
	int y_min_1 = ((y - 1) + height) % height;
	grid_[0] = grid_const.grid[x_min_1*width + y_min_1];

	//récupérer les voisins sur grid start
	size_t cells_alive = grid_[0] + grid_[1] + grid_[2] +
		grid_[3] + grid_[5] +
		grid_[6] + grid_[7] + grid_[8];

	grid_computed.grid[x*width + y] =
		(grid_[x*width + y]) //alive
		? (
			(cells_alive < 2 || cells_alive > 3)
			? false //kill
			: grid_[x*width + y] //forward
		)
			: ( // dead
			(cells_alive == 3)
			? true //resurect
			: grid_[x*width + y] //forward
		);
	
	//__syncthreads();
}


__global__ void gol_step_kernel_gl(const Grid grid_const, Grid grid_computed, unsigned char* colorBuffer, const char& color_true, const char& color_false){

	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	size_t width = blockDim.x * gridDim.x;
	//size_t height = blockDim.y * gridDim.y;

	//récupérer les voisins sur grid start
	size_t cells_alive = countAliveNeighbours(x, y, grid_const);

	grid_computed.grid[x*width + y] =
		(grid_const.grid[x*width + y]) //alive
		? (
			(cells_alive < 2 || cells_alive > 3)
			? false //kill
		: grid_const.grid[x*width + y] //forward
		)
		: ( // dead
			(cells_alive == 3)
			? true //resurect
		: grid_const.grid[x*width + y] //forward
		);

	colorBuffer[x*width + y] = grid_computed.grid[x*width + y] ? color_true : color_false;
}

__host__ void printGrid(Grid& grid){
	fprintf(stdout, "Grille %dx%d\n", grid.width, grid.height);
	for (size_t i = 0; i < grid.width; ++i){
		for (size_t j = 0; j < grid.height; ++j){
			fprintf(stdout, grid.grid[i*grid.width + j] ? "O" : "X");
		}
		fprintf(stdout, "\n");
	}
}

__host__ void do_step_gl(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed, unsigned char* colorBuffer, const char& color_true, const char& color_false){
	gol_step_kernel_gl <<< grid_size, block_size >>> (grid_const, grid_computed, colorBuffer, color_true, color_false);
	CudaCheckError();
	auto tmp = grid_computed;
	grid_computed = grid_const;
	grid_const = tmp;
}

__host__ void do_step(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed){
	gol_step_kernel_shared <<< grid_size, block_size, block_size.x*block_size.y*sizeof(bool) >>> (grid_const, grid_computed);
	CudaCheckError();
	auto tmp = grid_computed;
	grid_computed = grid_const;
	grid_const = tmp;
}

__host__ void launch_kernel(const Grid& cpu_grid, size_t nb_loop, unsigned int width, unsigned int height){

	//Affichage
	//printGrid(cpu_grid);

	Grid grid_const;
	Grid grid_computed;
	initGridCuda(grid_const, width, height);
	initGridCuda(grid_computed, width, height);

	CudaSafeCall(
		cudaMemcpy(grid_const.grid,	cpu_grid.grid,
		grid_const.width*grid_const.height*sizeof(bool),
		cudaMemcpyHostToDevice));

	dim3 grid_size = dim3(width / 8, height / 8);
	dim3 block_size = dim3(8, 8);

	for (size_t i = 0; i < nb_loop; ++i){
		do_step(grid_size, block_size, grid_const, grid_computed);
	}

	CudaSafeCall(
		cudaMemcpy(cpu_grid.grid, grid_const.grid,
		cpu_grid.width*cpu_grid.height*sizeof(bool),
		cudaMemcpyDeviceToHost));

	//Affichage
	//printGrid(cpu_grid);

	freeGridCuda(grid_const);
	freeGridCuda(grid_computed);
}
