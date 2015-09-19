#pragma once

#include "coord2D.h"

class Cell{
public:
	enum State{
		Alive,
		Dead
	};

	Cell() : state_(Alive)
	{}

	Cell(State state, const Coord2D& coord2D) : state_(state)
	{}

	Cell(State state, size_t x, size_t y) : state_(state)
	{}

	bool isAlive() const { return state_ == Alive; }
	void kill() { state_ = Dead; }

	bool isDead() const { return !isAlive(); }
	void resurect() { state_ = Alive; }

	void go_on() const {}
	void go_on(const Cell& cell) { state_ = cell.state_; }

private:
	State state_;
};