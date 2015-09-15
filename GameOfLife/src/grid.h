#pragma once

#include "matrix.h"
#include "cell.h"

class Grid{
public:
	Grid() :
		grid_(0, 0) //Grille vide
	{}

	Grid(size_t width, size_t height) :
		grid_(width, height)
	{}

	Cell const& operator()(size_t x, size_t y) const{
		//Wrapping grid
		return grid_(x % grid_.width(), y % grid_.height());
	}

	Cell& operator()(size_t x, size_t y){
		//Wrapping grid
		return grid_(x % grid_.width(), y % grid_.height());
	}

	//Retour par copie.
	//On ne veut pas que les état change quand une modif est faite dans un autre thread
	std::vector<Cell> getNeighbouringCells(size_t x, size_t y) const{
		auto neighbouring_cells = std::vector<Cell>{};
		for (size_t i = x - 1; i < x + 2; ++i){
			for (size_t j = y - 1; j < y + 2; ++j){
				neighbouring_cells.push_back(this->operator()(i, j));
			}
		}
		return neighbouring_cells;
	}

	size_t width() const { return grid_.width(); }
	size_t height() const { return grid_.height(); }

private:
	Matrix<Cell> grid_;
};