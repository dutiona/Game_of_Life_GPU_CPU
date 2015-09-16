#include "gol_kernel.h"
#include "Display.h"

#include <iostream>
#include <limits>
#include <chrono>

int main(int argc, char* argv[]){

	//GLDisplay::init(&argc, argv);
	//GLDisplay::run();

	size_t nb_loop = 10000;
	unsigned int width = 2 * 2 * 2 * 2 * 2 * 2 * 2; //2^7
	unsigned int height = 2 * 2 * 2 * 2 * 2 * 2 * 2; //2^7
	int fill_thresold = 30;
	Grid cpu_grid;
	initGrid(cpu_grid, width, height);

	//Random init
	const auto seed = std::random_device{}(); //seed ne dépend pas de std::chrono
	std::mt19937 rd_mt_engine(seed); // mt19937 est le mersenne_twister_engine standard
	std::uniform_int_distribution<int> uniform_distrib(1, 100); // distribution 1 à 100 uniforme
	for (unsigned int i = 0; i < width; ++i){
		for (unsigned int j = 0; j < height; ++j){
			//Remplissage aléatoire de la grille en fonction du fill_thresold
			cpu_grid.grid[i*cpu_grid.width + j] = uniform_distrib(rd_mt_engine) < fill_thresold;
		}
	}

	const auto start_serial = std::chrono::high_resolution_clock::now();
	launch_kernel(cpu_grid, nb_loop, width, height);
	const auto elapsed_serial = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_serial)).count());
	std::cout << elapsed_serial << "ms" << std::endl;

	freeGrid(cpu_grid);

	//Pause
	std::cout << "Entrez sur enter pour continuer..." << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	return EXIT_SUCCESS;
}