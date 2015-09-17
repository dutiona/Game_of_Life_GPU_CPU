#include "gol_kernel.cuh"
#include "Display.h"

#include <iostream>
#include <limits>
#include <chrono>

int main(int argc, char* argv[]){

	//GLDisplay::init(&argc, argv);
	//GLDisplay::run();

	size_t nb_loop = 10000;
	unsigned int width = 16 * 8 * 2; //2^7
	unsigned int height = 16 * 8 * 2; //2^7
	int fill_thresold = 30;
	Grid cpu_grid;

	//Global
	
	initGrid(cpu_grid, width, height);
	//Random init
	const auto seed = std::random_device{}(); //seed ne d�pend pas de std::chrono
	auto rd_mt_engine = std::mt19937{ seed }; // mt19937 est le mersenne_twister_engine standard
	auto uniform_distrib = std::uniform_int_distribution<int>{1, 100}; // distribution 1 � 100 uniforme
	for (unsigned int i = 0; i < width; ++i){
		for (unsigned int j = 0; j < height; ++j){
			//Remplissage al�atoire de la grille en fonction du fill_thresold
			cpu_grid.grid[i*cpu_grid.width + j] = uniform_distrib(rd_mt_engine) < fill_thresold;
		}
	}
	const auto start_global = std::chrono::high_resolution_clock::now();
	launch_kernel(cpu_grid, nb_loop, width, height);
	const auto elapsed_global = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_global)).count());
	std::cout << elapsed_global << "ms" << std::endl;
	freeGrid(cpu_grid);

	//Shared

	initGrid(cpu_grid, width, height);
	//Random init
	rd_mt_engine = std::mt19937{ seed }; // same seed
	for (unsigned int i = 0; i < width; ++i){
		for (unsigned int j = 0; j < height; ++j){
			//Remplissage al�atoire de la grille en fonction du fill_thresold
			cpu_grid.grid[i*cpu_grid.width + j] = uniform_distrib(rd_mt_engine) < fill_thresold;
		}
	}
	const auto start_shared = std::chrono::high_resolution_clock::now();
	launch_kernel_shared(cpu_grid, nb_loop, width, height);
	const auto elapsed_shared = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_shared)).count());
	std::cout << elapsed_shared << "ms" << std::endl;

	std::cout << "Accelerating factor : " << (elapsed_global / elapsed_shared) << std::endl;

	//Pause
	std::cout << "Entrez sur enter pour continuer..." << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	return EXIT_SUCCESS;
}