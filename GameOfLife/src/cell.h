#pragma once

#include "coord2D.h"

class Cell{
public:
	enum State{
		Alive,
		Dead
	};

	Cell() :
		state_(Alive),
		coord2D_(Coord2D::OXY)
	{}

	Cell(State state, const Coord2D& coord2D) :
		state_(state),
		coord2D_(coord2D)
	{}

	Cell(State state, size_t x, size_t y) :
		state_(state),
		coord2D_({ x, y })
	{}

	bool isAlive() const { return state_ == Alive; }
	void kill() { state_ = Dead; }

	bool isDead() const { return !isAlive(); }
	void resurect() { state_ = Alive; }

	void go_on() const {}
	void go_on(const Cell& cell) { state_ = cell.state_; }

	const Coord2D& getCoord() const { return coord2D_; }
	Coord2D& coord(){ return coord2D_; }

private:
	State state_;
	Coord2D coord2D_;
};