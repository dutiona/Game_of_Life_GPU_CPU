#include <iostream>
#include <string>
#include <limits>

#include "GoL_Engine.h"

int main(int /*argc*/, char* /*argv[]*/){

	auto gol_engine = GoL_Engine{ 30, 30, 30, 1000 };
	gol_engine.init();

	std::cout << gol_engine;

	gol_engine.run();

	std::cout << gol_engine;

	//Pause
	std::cout << "Entrez sur enter pour continuer..." << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	return EXIT_SUCCESS;
}