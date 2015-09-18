#include "gol_kernel.cuh"

#include <Windows.h>
#include <time.h>

//Internal

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

__host__ void printGrid(Grid& grid){
	fprintf(stdout, "Grille %dx%d\n", grid.width, grid.height);
	for (size_t i = 0; i < grid.width; ++i){
		for (size_t j = 0; j < grid.height; ++j){
			fprintf(stdout, grid.grid[i*grid.width + j] ? "O" : "X");
		}
		fprintf(stdout, "\n");
	}
}

__host__ bool gridAreEquals(const Grid& glhs, const Grid& grhs){
	for (int i = 0; i < glhs.width; ++i){
		for (int j = 0; j < grhs.height; ++j){
			if (glhs.grid[i*glhs.width + j] != grhs.grid[i*glhs.width + j]){
				return false;
			}
		}
	}
	return true;
}


//Global


__global__ void gol_step_kernel(Grid grid_const, Grid grid_computed){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int width = blockDim.x * gridDim.x;
	int height = blockDim.y * gridDim.y;

	//récupérer les voisins sur grid start
	int x_min_1 = ((x - 1) + width) % width;
	int y_min_1 = ((y - 1) + height) % height;
	int x_plus_1 = ((x + 1) + width) % width;
	int y_plus_1 = ((y + 1) + height) % height;
	size_t cells_alive =
		grid_const.grid[x_min_1*width + y_min_1] + grid_const.grid[x_min_1*width + y] + grid_const.grid[x_min_1*width + y_plus_1] +
		grid_const.grid[x*width + y_min_1] + grid_const.grid[x*width + y_plus_1] +
		grid_const.grid[x_plus_1*width + y_min_1] + grid_const.grid[x_plus_1*width + y] + grid_const.grid[x_plus_1*width + y_plus_1];

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

__host__ void do_step(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed){
	gol_step_kernel <<< grid_size, block_size >>> (grid_const, grid_computed);
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
		cudaMemcpy(grid_const.grid, cpu_grid.grid,
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


//Shared

__global__ void gol_step_kernel_shared(const Grid grid_const, Grid grid_computed){
	extern __shared__ bool grid_[];

	//Taille de la mémoire globale
	int width = (blockDim.x - 2) * gridDim.x;
	int height = (blockDim.y - 2) * gridDim.y;

	//Coordonnées dans la grid memory shared
	int x = threadIdx.x;
	int y = threadIdx.y;
	int pnt = x*blockDim.y + y;

	//Calcul de la correspondance dans la global
	//On fait -1 sur x,y pour être en négatif sur les bords puis on réaditionne et on module
	int x_from = (blockIdx.x*(blockDim.x - 2) + x - 1 + width) % width;
	int y_from = (blockIdx.y*(blockDim.y - 2) + y - 1 + height) % height;
	int pnt_from = x_from*height + y_from;

	//On charge la donnée dans la shared
	grid_[pnt] = grid_const.grid[pnt_from];

	//On charge le block dans la shared
	__syncthreads();

	//Ces thread sont IDLE, ils ne servent qu'à charger la shared pour la  lecture
	if (x != 0 && y != 0 && x != blockDim.x - 1 && y != blockDim.y - 1){

		//récupérer les voisins sur grid start
		size_t cells_alive =
			grid_[(x - 1)*blockDim.y + (y - 1)] + grid_[(x - 1)*blockDim.y + y] + grid_[(x - 1)*blockDim.y + (y + 1)] +
			grid_[x*blockDim.y + (y - 1)] + grid_[x*blockDim.y + (y + 1)] +
			grid_[(x + 1)*blockDim.y + (y - 1)] + grid_[(x + 1)*blockDim.y + y] + grid_[(x + 1)*blockDim.y + (y + 1)];

		grid_computed.grid[pnt_from] =
			(grid_[pnt]) //alive
			? (
				(cells_alive < 2 || cells_alive > 3)
				? false //kill
				: grid_[pnt] //forward
			)
			: ( // dead
				(cells_alive == 3)
				? true //resurect
				: grid_[pnt] //forward
			);
	}
}

__host__ void do_step_shared(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed){
	dim3 block_size_extended = dim3(block_size.x + 2, block_size.y + 2);
	gol_step_kernel_shared <<< grid_size, block_size_extended, (block_size_extended.x) * (block_size_extended.y) * sizeof(bool) >>> (grid_const, grid_computed);
	CudaCheckError();
	auto tmp = grid_computed;
	grid_computed = grid_const;
	grid_const = tmp;
}

__host__ void launch_kernel_shared(const Grid& cpu_grid, size_t nb_loop, unsigned int width, unsigned int height){

	//Affichage
	//printGrid(cpu_grid);

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

	for (size_t i = 0; i < nb_loop; ++i){
		do_step_shared(grid_size, block_size, grid_const, grid_computed);
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



//OpenGL


__global__ void gol_step_kernel_shared_gl(const Grid grid_const, Grid grid_computed, uchar4* grid_pixels, uchar4 color_true, uchar4 color_false){
	extern __shared__ bool grid_[];

	//Taille de la mémoire globale
	int width = (blockDim.x - 2) * gridDim.x;
	int height = (blockDim.y - 2) * gridDim.y;

	//Coordonnées dans la grid memory shared
	int x = threadIdx.x;
	int y = threadIdx.y;
	int pnt = x*blockDim.y + y;

	//Calcul de la correspondance dans la global
	//On fait -1 sur x,y pour être en négatif sur les bords puis on réaditionne et on module
	int x_from = (blockIdx.x*(blockDim.x - 2) + x - 1 + width) % width;
	int y_from = (blockIdx.y*(blockDim.y - 2) + y - 1 + height) % height;
	int pnt_from = x_from*height + y_from;

	//On charge la donnée dans la shared
	grid_[pnt] = grid_const.grid[pnt_from];

	//On charge le block dans la shared
	__syncthreads();

	//Ces thread sont IDLE, ils ne servent qu'à charger la shared pour la  lecture
	if (x != 0 && y != 0 && x != blockDim.x - 1 && y != blockDim.y - 1){

		//récupérer les voisins sur grid start
		size_t cells_alive =
			grid_[(x - 1)*blockDim.y + (y - 1)] + grid_[(x - 1)*blockDim.y + y] + grid_[(x - 1)*blockDim.y + (y + 1)] +
			grid_[x*blockDim.y + (y - 1)] + grid_[x*blockDim.y + (y + 1)] +
			grid_[(x + 1)*blockDim.y + (y - 1)] + grid_[(x + 1)*blockDim.y + y] + grid_[(x + 1)*blockDim.y + (y + 1)];

		bool state =
			(grid_[pnt]) //alive
			? (
				(cells_alive < 2 || cells_alive > 3)
				? false //kill
				: grid_[pnt] //forward
			)
			: ( // dead
				(cells_alive == 3)
				? true //resurect
				: grid_[pnt] //forward
			);

		grid_computed.grid[pnt_from] = state;
		grid_pixels[pnt_from] = state ? color_true : color_false;
	}
}

__host__ void do_step_shared_gl(const dim3& grid_size, const dim3& block_size, Grid& grid_const, Grid& grid_computed, uchar4* grid_pixels, uchar4 color_true, uchar4 color_false){
	dim3 block_size_extended = dim3(block_size.x + 2, block_size.y + 2);
	gol_step_kernel_shared_gl <<< grid_size, block_size_extended, (block_size_extended.x) * (block_size_extended.y) * sizeof(bool) >>> (grid_const, grid_computed, grid_pixels, color_true, color_false);
	CudaCheckError();
	auto tmp = grid_computed;
	grid_computed = grid_const;
	grid_const = tmp;
}