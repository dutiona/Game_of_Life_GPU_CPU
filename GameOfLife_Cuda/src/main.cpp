#include "gol_kernel.h"
#include "Display.h"

#include <iostream>
#include <limits>

int main(int /*argc*/, char* /*argv[]*/){



	//Pause
	std::cout << "Entrez sur enter pour continuer..." << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	return EXIT_SUCCESS;
}