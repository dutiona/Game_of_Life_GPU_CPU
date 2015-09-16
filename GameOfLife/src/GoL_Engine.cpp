#include "GoL_Engine.h"

#include <cassert>
#include <random>
#include <functional>
#include <thread>
#include <omp.h>

namespace{
	inline size_t countAliveNeighbours(size_t x, size_t y, const Grid& grid){
		return grid(x - 1, y - 1).isAlive() + grid(x - 1, y).isAlive() + grid(x - 1, y + 1).isAlive() +
			grid(x, y - 1).isAlive() + grid(x, y + 1).isAlive() +
			grid(x + 1, y - 1).isAlive() + grid(x + 1, y).isAlive() + grid(x + 1, y + 1).isAlive();
	}

	inline size_t countDeadNeighbours(size_t x, size_t y, const Grid& grid){
		return 8 - countAliveNeighbours(x, y, grid);
	}
}

GoL_Engine::GoL_Engine(size_t width_pow2, size_t height_pow2, size_t fill_thresold) :
grid_(std::pow(2, width_pow2), std::pow(2, height_pow2)),
fill_thresold_(fill_thresold),
step_number_(0),
max_step_(std::numeric_limits<size_t>::max())
{}

GoL_Engine::GoL_Engine(size_t width_pow2, size_t height_pow2, size_t fill_thresold, size_t max_step) :
grid_(std::pow(2, width_pow2), std::pow(2, height_pow2)),
fill_thresold_(fill_thresold),
step_number_(0),
max_step_(max_step)
{}

void GoL_Engine::registerObserver(std::shared_ptr<const IObserver> o){
	observers_.push_back(o);
}

void GoL_Engine::init(){
	const auto seed = std::random_device{}(); //seed ne dépend pas de std::chrono
	std::mt19937 rd_mt_engine(seed); // mt19937 est le mersenne_twister_engine standard
	std::uniform_int_distribution<int> uniform_distrib(1, 100); // distribution 1 à 100 uniforme

	for (size_t i = 0; i < grid_.width(); ++i){
		for (size_t j = 0; j < grid_.height(); ++j){
			//Remplissage aléatoire de la grille en fonction du fill_thresold
			if (uniform_distrib(rd_mt_engine) < fill_thresold_){
				grid_(i, j).resurect();
			}
			else{
				grid_(i, j).kill();
			}
		}
	}
}

const Grid* GoL_Engine::do_step(){
	if (step_number_ < max_step_){
		auto grid_working_cpy = Grid{ grid_.width(), grid_.height() }; //Copie de travail

		for (size_t i = 0; i < grid_.width(); ++i){
			for (size_t j = 0; j < grid_.height(); ++j){
				const auto cells_alive = ::countAliveNeighbours(i, j, grid_);
				//Any live cell with fewer than two live neighbors dies, as if caused by under - population.
				//Any live cell with more than three live neighbors dies, as if by overcrowding.
				if (grid_(i, j).isAlive() && (cells_alive < 2 || cells_alive > 3)){
					grid_working_cpy(i, j).kill();
				}
				//Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
				else if (grid_(i, j).isDead() && cells_alive == 3){
					grid_working_cpy(i, j).resurect();
				}
				//Any live cell with two or three live neighbors lives on to the next generation.
				else{
					grid_working_cpy(i, j).go_on(grid_(i, j));
				}
			}
		}

		std::swap(grid_, grid_working_cpy);

		++step_number_;
		notifyObservers();
		return &grid_;
	}
	else{
		notifyObservers();
		return nullptr;
	}
}


const Grid* GoL_Engine::do_step_omp(){
	if (step_number_ < max_step_){
		auto grid_working_cpy = Grid{ grid_.width(), grid_.height() }; //Copie de travail

		int i, j;
#pragma omp parallel for private(i)
		for (i = 0; i < grid_.width(); ++i){
#pragma omp parallel for private(j)
			for (j = 0; j < grid_.height(); ++j){
				const auto cells_alive = ::countAliveNeighbours(i, j, grid_);
				//Any live cell with fewer than two live neighbors dies, as if caused by under - population.
				//Any live cell with more than three live neighbors dies, as if by overcrowding.
				if (grid_(i, j).isAlive() && (cells_alive < 2 || cells_alive > 3)){
					grid_working_cpy(i, j).kill();
				}
				//Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
				else if (grid_(i, j).isDead() && cells_alive == 3){
					grid_working_cpy(i, j).resurect();
				}
				//Any live cell with two or three live neighbors lives on to the next generation.
				else{
					grid_working_cpy(i, j).go_on(grid_(i, j));
				}
			}
		}

		std::swap(grid_, grid_working_cpy);

		++step_number_;
		notifyObservers();
		return &grid_;
	}
	else{
		notifyObservers();
		return nullptr;
	}
}

const Grid* GoL_Engine::do_step_std_thread(size_t nb_thread){
	if (step_number_ < max_step_){
		auto grid_working_cpy = Grid{ grid_.width(), grid_.height() }; //Copie de travail

		const auto thread_functor_factory = [&](size_t start_width, size_t end_width, size_t start_height, size_t end_height) -> std::function<void(void)>{
			return [&](){
				for (size_t i = start_width; i < end_width && i < grid_.width(); ++i){
					for (size_t j = start_height; j < end_height && j < grid_.height(); ++j){
						const auto cells_alive = ::countAliveNeighbours(i, j, grid_);
						//Any live cell with fewer than two live neighbors dies, as if caused by under - population.
						//Any live cell with more than three live neighbors dies, as if by overcrowding.
						if (grid_(i, j).isAlive() && (cells_alive < 2 || cells_alive > 3)){
							grid_working_cpy(i, j).kill();
						}
						//Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
						else if (grid_(i, j).isDead() && cells_alive == 3){
							grid_working_cpy(i, j).resurect();
						}
						//Any live cell with two or three live neighbors lives on to the next generation.
						else{
							grid_working_cpy(i, j).go_on(grid_(i, j));
						}
					}
				}
			};
		};

		const size_t delta_width = grid_.width() / (std::sqrt(nb_thread));
		const size_t delta_height = grid_.height() / (std::sqrt(nb_thread));
		std::vector<std::thread> thread_list;

		for (size_t i = 0; i < grid_.width(); i += delta_width){
			for (size_t j = 0; j < grid_.height(); j += delta_height){
				thread_list.emplace_back(std::move(std::thread{ thread_functor_factory(i, i + delta_width, j, j + delta_height) }));
			}
		}

		for (auto& thread : thread_list){
			thread.join();
		}

		std::swap(grid_, grid_working_cpy);

		++step_number_;
		notifyObservers();
		return &grid_;
	}
	else{
		notifyObservers();
		return nullptr;
	}
}

void GoL_Engine::run_serial(){
	while (do_step() != nullptr);
}

void GoL_Engine::run_omp(size_t nb_thread){
	omp_set_num_threads(nb_thread);
	while (do_step_omp() != nullptr);
}

void GoL_Engine::run_std_thread(size_t nb_thread){
	while (do_step_std_thread(nb_thread) != nullptr);
}


bool GoL_Engine::allAlive() const{
	for (size_t i = 0; i < grid_.width(); ++i){
		for (size_t j = 0; j < grid_.height(); ++j){
			if (grid_(i, j).isDead())
				return false;
		}
	}
	return true;
}

bool GoL_Engine::allDead() const{
	for (size_t i = 0; i < grid_.width(); ++i){
		for (size_t j = 0; j < grid_.height(); ++j){
			if (grid_(i, j).isAlive())
				return false;
		}
	}
	return true;
}

const Grid& GoL_Engine::getGrid() const{
	return grid_;
}

void GoL_Engine::notifyObservers(){
	for (const auto& o : observers_){
		o->update(grid_);
	}
}

std::ostream& operator<<(std::ostream& oss, const GoL_Engine& gol_engine) {
	auto grid = gol_engine.getGrid();
	oss << "Grille " << grid.width() << "x" << grid.height() << std::endl;
	oss << "----" << std::endl;
	for (size_t i = 0; i < grid.width(); ++i){
		for (size_t j = 0; j < grid.height(); ++j){
			oss << (grid(i, j).isAlive() ? "O" : "X");
		}
		oss << std::endl;
	}
	return oss;
}