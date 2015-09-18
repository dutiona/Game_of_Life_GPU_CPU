#include <iostream>
#include <string>
#include <limits>
#include <memory>
#include <chrono>

#include "GoL_Engine.h"

int main(int /*argc*/, char* /*argv[]*/){

	//Serial

	std::cout << "GoL serial :" << std::endl;
	auto gol_engine = std::make_unique<GoL_Engine>(10, 10, 70, 1000); //2^7
	gol_engine->init();

	const auto start_serial = std::chrono::high_resolution_clock::now();
	gol_engine->run_serial();
	const auto elapsed_serial = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_serial)).count());
	std::cout << elapsed_serial << "ms" << std::endl;



	//Omp

	std::cout << "GoL OpenMP :" << std::endl;
	gol_engine = std::make_unique<GoL_Engine>(10, 10, 70, 1000); //2^7
	gol_engine->init();

	const auto start_omp = std::chrono::high_resolution_clock::now();
	gol_engine->run_omp(8);
	const auto elapsed_omp = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_omp)).count());
	std::cout << elapsed_omp << "ms" << std::endl;
	std::cout << "Accelerating factor : " << (elapsed_serial / elapsed_omp) << std::endl;



	//std::thread

	std::cout << "GoL std::thread :" << std::endl;
	gol_engine = std::make_unique<GoL_Engine>(10, 10, 70, 1000); //2^7
	gol_engine->init();

	const auto start_std_thread = std::chrono::high_resolution_clock::now();
	gol_engine->run_std_thread(8);
	const auto elapsed_std_thread = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::high_resolution_clock::now() - start_std_thread)).count());
	std::cout << elapsed_std_thread << "ms" << std::endl;
	std::cout << "Accelerating factor : " << (elapsed_serial / elapsed_std_thread) << std::endl;
	//Les perfs sont mauvaises car on spawn un thread et on le kill pour chaque génération. Il faudrait garder les std::thread et faire un système de task dans une Queue.

	//Pause
	std::cout << "Entrez sur enter pour continuer..." << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	return EXIT_SUCCESS;
}