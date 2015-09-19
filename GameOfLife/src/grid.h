#pragma once

#include "matrix.h"
#include "cell.h"

class Grid{
public:
	Grid() :
		grid_(0, 0) //Grille vide
	{}

	Grid(int width, int height) :
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

	int width() const { return grid_.width(); }
	int height() const { return grid_.height(); }

private:
	Matrix<Cell> grid_;
};