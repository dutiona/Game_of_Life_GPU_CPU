#include <iostream>
#include <string>
#include <limits>
#include <memory>
#include <chrono>

#include "GoL_Engine.h"

int main(int /*argc*/, char* /*argv[]*/){

	auto gol_engine = std::make_unique<GoL_Engine>(7, 7, 30, 1000); //2^6 = 64
	gol_engine->init();

	auto start = std::chrono::high_resolution_clock::now();
	//std::cout << *gol_engine;
	gol_engine->run(false);
	//std::cout << *gol_engine;
	auto elapsed_serial = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start)).count());
	std::cout << elapsed_serial << "ms" << std::endl;

	gol_engine = std::make_unique<GoL_Engine>(7, 7, 30, 1000); //2^6 = 64
	gol_engine->init();

	start = std::chrono::high_resolution_clock::now();
	//std::cout << *gol_engine;
	gol_engine->run(true);
	//std::cout << *gol_engine;
	auto elapsed_omp = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start)).count());
	std::cout << elapsed_omp << "ms" << std::endl;

	std::cout << "Accelerating factor : " << (elapsed_serial / elapsed_omp) << std::endl;

	//Pause
	std::cout << "Entrez sur enter pour continuer..." << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	return EXIT_SUCCESS;
}