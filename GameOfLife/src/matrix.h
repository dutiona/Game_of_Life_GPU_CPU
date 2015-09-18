#pragma once

#include <vector>
#include <cassert>

template<class T>
class Matrix{
public:
	Matrix(int width, int height, T const& t = T{}) :
		w(width),
		h(height),
		datas(w*h, t)
	{}

	Matrix(const Matrix& m) = default;

	Matrix& operator=(Matrix const& m) = default;

	T const& operator()(int x, int y) const{
		assert(x < w && y < h && "Out of bound");
		return datas[y*w + x];
	}

	T& operator()(int x, int y){
		assert(x < w && y < h && "Out of bound");
		return datas[y*w + x];
	}

	int width() const{ return w; }
	int height() const{ return h; }

private:
	int w;
	int h;
	std::vector<T> datas;
};
