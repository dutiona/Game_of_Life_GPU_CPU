#include "gol_kernel.h"

#include <time.h>

__host__ void initGrid(Grid& g, size_t w, size_t h){
	g.width = w;
	g.height = h;
	g.grid = (bool*)malloc(w*h*sizeof(bool));
}
__host__ void initGridCuda(Grid& g, size_t w, size_t h){
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

__device__ inline size_t countAliveNeighbours(size_t x, size_t y, const Grid& g){
	size_t width = blockDim.x * gridDim.x;
	size_t height = blockDim.y * gridDim.y;
	int x_min_1 = ((x - 1) + width) % width;
	int x_plus_1 = ((x + 1) + width) % width;
	int y_min_1 = ((y - 1) + height) % height;
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

__host__ void printGrid(Grid& grid){
	fprintf(stdout, "Grille %dx%d\n", grid.width, grid.height);
	for (size_t i = 0; i < grid.width; ++i){
		for (size_t j = 0; j < grid.height; ++j){
			fprintf(stdout, grid.grid[i*grid.width + j] ? "O" : "X");
		}
		fprintf(stdout, "\n");
	}
}

__host__ const Grid* do_step(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed){
	gol_step_kernel <<< grid_size, block_size >>> (grid_const, grid_computed);
	CudaCheckError();
	auto tmp = grid_computed;
	grid_computed = grid_const;
	grid_const = tmp;
	return &grid_const;
}

int main(){
	clock_t begin, end;
	double time_spent;

	size_t nb_loop = 10000;
	size_t width = 2 * 2 * 2 * 2 * 2 * 2 * 2; //2^7
	size_t height = 2 * 2 * 2 * 2 * 2 * 2 * 2; //2^7
	size_t fill_thresold = 30;

	Grid cpu_grid;
	initGrid(cpu_grid, width, height);

	//Random init
	srand(time(NULL));
	for (size_t i = 0; i < cpu_grid.width; ++i){
		for (size_t j = 0; j < cpu_grid.height; ++j){
			cpu_grid.grid[i*cpu_grid.width + j] = (rand() % 100 < fill_thresold);
		}
	}

	//Affichage
	//printGrid(cpu_grid);

	Grid grid_const;
	Grid grid_computed;
	initGridCuda(grid_const, width, height);
	initGridCuda(grid_computed, width, height);

	//Start chrono
	begin = clock();

	CudaSafeCall(
		cudaMemcpy(grid_const.grid,	cpu_grid.grid,
		grid_const.width*grid_const.height*sizeof(bool),
		cudaMemcpyHostToDevice));

	dim3 grid_size = dim3(width / 8, height / 8);
	dim3 block_size = dim3(8, 8);

	for (int i = 0; i < nb_loop; ++i){
		do_step(grid_size, block_size, grid_const, grid_computed);
	}

	CudaSafeCall(
		cudaMemcpy(cpu_grid.grid, grid_const.grid,
		cpu_grid.width*cpu_grid.height*sizeof(bool),
		cudaMemcpyDeviceToHost));

	end = clock();
	time_spent = ((double)(end - begin) / CLOCKS_PER_SEC)*1000;
	fprintf(stdout, "Execution time : %fms\n", time_spent);

	//Affichage
	//printGrid(cpu_grid);

	freeGrid(cpu_grid);
	freeGridCuda(grid_const);
	freeGridCuda(grid_computed);


	system("PAUSE");


	return EXIT_SUCCESS;
}
