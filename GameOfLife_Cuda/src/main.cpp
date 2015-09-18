#include "gol_kernel.cuh"
#include "Display.h"

#include <iostream>
#include <limits>
#include <chrono>
#include <cassert>
#include <random>

int main(int argc, char* argv[]){

	//true : affichage openGL, false profiling des kernels
	const bool openGL = true;

	if (openGL){
		GLDisplay::init(&argc, argv);
		GLDisplay::run();
	}
	else{

		size_t nb_loop = 1000; //Nombre de loop dans la simulation
		unsigned int width = 8 * 8 * 8 * 2; //2^10
		unsigned int height = 8 * 8 * 8 * 2; //2^10
		int fill_thresold = 70;
		Grid cpu_grid_global, cpu_grid_shared;

		//Init
		initGrid(cpu_grid_global, width, height);
		initGrid(cpu_grid_shared, width, height);

		const auto seed = std::random_device{}(); //seed ne dépend pas de std::chrono
		auto rd_mt_engine = std::mt19937{ seed }; // mt19937 est le mersenne_twister_engine standard
		auto uniform_distrib = std::uniform_int_distribution<int>{1, 100}; // distribution 1 à 100 uniforme
		for (unsigned int i = 0; i < width; ++i){
			for (unsigned int j = 0; j < height; ++j){
				//Remplissage aléatoire de la grille en fonction du fill_thresold
				const auto v = uniform_distrib(rd_mt_engine) < (100 - fill_thresold);
				cpu_grid_global.grid[i*cpu_grid_global.width + j] = v;
				cpu_grid_shared.grid[i*cpu_grid_shared.width + j] = v;
			}
		}

		//Même distribution dans les grilles
		assert(gridAreEquals(cpu_grid_global, cpu_grid_shared));

		//Global
		const auto start_global = std::chrono::high_resolution_clock::now();
		launch_kernel(cpu_grid_global, nb_loop, width, height);
		const auto elapsed_global = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_global)).count());
		std::cout << "GoL cuda global : " << elapsed_global << "ms" << std::endl;


		//Shared
		const auto start_shared = std::chrono::high_resolution_clock::now();
		launch_kernel_shared(cpu_grid_shared, nb_loop, width, height);
		const auto elapsed_shared = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_shared)).count());
		std::cout << "GoL cuda shared : " << elapsed_shared << "ms" << std::endl;


		std::cout << "Accelerating factor : " << (elapsed_global / elapsed_shared) << std::endl;

		//Vérification résultat identique.
		assert(gridAreEquals(cpu_grid_global, cpu_grid_shared));

		//Free
		freeGrid(cpu_grid_global);
		freeGrid(cpu_grid_shared);

		//Pause
		std::cout << "Entrez sur enter pour continuer..." << std::endl;
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

	return EXIT_SUCCESS;
}