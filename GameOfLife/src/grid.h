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

	Cell const& operator()(int x, int y) const{
		//Wrapping grid
		assert((x + grid_.width()) % grid_.width() >= 0 && (y + grid_.height()) % grid_.height() >= 0);
		return grid_((x + grid_.width()) % grid_.width(), (y + grid_.height()) % grid_.height());
	}

	Cell& operator()(int x, int y){
		//Wrapping grid
		assert((x + grid_.width()) % grid_.width() >= 0 && (y + grid_.height()) % grid_.height() >= 0);
		return grid_((x + grid_.width()) % grid_.width(), (y + grid_.height()) % grid_.height());
	}

	//Retour par référence.
	//On ne veut pas que les état change quand une modif est faite dans un autre thread
	std::vector<const Cell*> getNeighbouringCells(int x, int y) const{
		auto neighbouring_cells = std::vector<const Cell*>{};
		for (int i = x - 1; i < x + 2; ++i){
			for (int j = y - 1; j < y + 2; ++j){
				neighbouring_cells.push_back(&this->operator()(i, j));
			}
		}
		return neighbouring_cells;
	}

	size_t width() const { return grid_.width(); }
	size_t height() const { return grid_.height(); }

private:
	Matrix<Cell> grid_;
};