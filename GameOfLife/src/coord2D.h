#pragma once

class Coord2D{
public:
	Coord2D() :
		x_(0),
		y_(0)
	{}

	Coord2D(size_t x, size_t y) :
		x_(x),
		y_(y)
	{}

	size_t getX() const { return x_; }
	size_t getY() const { return y_; }

	bool operator==(const Coord2D& r_c) const { return x_ == r_c.x_ && y_ == r_c.y_; }
	bool operator!=(const Coord2D& r_c) const { return !(*this == r_c); }

	static const Coord2D OXY;
private:

	size_t x_;
	size_t y_;
};