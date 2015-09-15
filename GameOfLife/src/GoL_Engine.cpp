#include "GoL_Engine.h"

#include <random>

namespace{
	size_t countAliveNeighbours(const std::vector<Cell>& pack){
		auto cell_alive = 0u;
		for (const auto& cell : pack){
			if (cell.isAlive())
				++cell_alive;
		}
		return cell_alive;
	}

	size_t countDeadNeighbours(const std::vector<Cell>& pack){
		return pack.size() - countAliveNeighbours(pack);
	}
}

GoL_Engine::GoL_Engine(size_t width, size_t height, size_t fill_thresold) :
grid_(width, height),
fill_thresold_(fill_thresold),
step_number_(0),
max_step_(std::numeric_limits<size_t>::max())
{}

GoL_Engine::GoL_Engine(size_t width, size_t height, size_t fill_thresold, size_t max_step) :
grid_(width, height),
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

bool GoL_Engine::do_step(){
	if (step_number_ < max_step_){
		kill_lowNeighbours();
		live_2to3neighbours();
		kill_highNeighbours();
		resurect_highNeighbours();
		++step_number_;
		if (step_number_ % 10 == 0){
			std::cout << step_number_ << std::endl;
		}
		return true;
	}
	else{
		return false;
	}
}

void GoL_Engine::run(){
	while (do_step())
		notifyObservers();

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

void GoL_Engine::kill_lowNeighbours(){
	for (size_t i = 0; i < grid_.width(); ++i){
		for (size_t j = 0; j < grid_.height(); ++j){
			//Any live cell with fewer than two live neighbors dies, as if caused by under - population.
			if (grid_(i, j).isAlive() && ::countAliveNeighbours(grid_.getNeighbouringCells(i, j)) < 2){
				grid_(i, j).kill();
			}
		}
	}
}

void GoL_Engine::live_2to3neighbours(){
	for (size_t i = 0; i < grid_.width(); ++i){
		for (size_t j = 0; j < grid_.height(); ++j){
			//Any live cell with two or three live neighbors lives on to the next generation.
			const auto cells_alive = ::countAliveNeighbours(grid_.getNeighbouringCells(i, j));
			if (grid_(i, j).isAlive() && (cells_alive == 2 || cells_alive == 3)){
				grid_(i, j).go_on();
			}
		}
	}
}

void GoL_Engine::kill_highNeighbours(){
	for (size_t i = 0; i < grid_.width(); ++i){
		for (size_t j = 0; j < grid_.height(); ++j){
			//Any live cell with more than three live neighbors dies, as if by overcrowding.
			if (grid_(i, j).isAlive() && ::countAliveNeighbours(grid_.getNeighbouringCells(i, j)) > 3){
				grid_(i, j).kill();
			}
		}
	}
}

void GoL_Engine::resurect_highNeighbours(){
	for (size_t i = 0; i < grid_.width(); ++i){
		for (size_t j = 0; j < grid_.height(); ++j){
			//Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
			if (grid_(i, j).isDead() && ::countAliveNeighbours(grid_.getNeighbouringCells(i, j)) == 3){
				grid_(i, j).resurect();
			}
		}
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