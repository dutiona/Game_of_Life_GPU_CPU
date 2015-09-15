#pragma once

#include <vector>
#include <cassert>

template<class T>
class Matrix{
public:
	Matrix(size_t width, size_t height, T const& t = T{}) :
		w(width),
		h(height),
		datas(w*h, t)
	{}

	Matrix(const Matrix& m) = default;

	Matrix& operator=(Matrix const& m) = default;

	T const& operator()(size_t x, size_t y) const{
		assert(x < w && y < h && "Out of bound");
		return datas[y*w + x];
	}

	T& operator()(size_t x, size_t y){
		assert(x < w && y < h && "Out of bound");
		return datas[y*w + x];
	}

	std::size_t width() const{ return w; }
	std::size_t height() const{ return h; }

private:
	std::size_t w;
	std::size_t h;
	std::vector<T> datas;
};
