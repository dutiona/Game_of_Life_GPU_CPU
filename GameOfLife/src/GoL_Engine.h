#pragma once

#include <limits>
#include <memory>
#include <iostream>

#include "grid.h"

class GoL_Engine{
public:

	//Prototype observer
	class IObserver{
	public:
		virtual void update(const Grid& grid) const = 0;
	};

	GoL_Engine(size_t width_pow2, size_t height_pow2, size_t fill_thresold);
	GoL_Engine(size_t width_pow2, size_t height_pow2, size_t fill_thresold, size_t max_step);

	void registerObserver(std::shared_ptr<const IObserver> o);

	void init();
	void run(bool multithreaded);

	bool allAlive() const;
	bool allDead() const;
	const Grid& getGrid() const;

private:

	bool do_step();
	bool do_step_multithreaded(size_t thread_number);
	bool do_step_multithreaded_omp(size_t thread_number);

	void kill_lowNeighbours();
	void live_2to3neighbours();
	void kill_highNeighbours();
	void resurect_highNeighbours();

	void notifyObservers();

	Grid grid_;
	size_t step_number_;
	size_t max_step_;

	const size_t fill_thresold_; //% de la grille initialis�e � Alive

	std::vector<std::shared_ptr<const IObserver>> observers_;
};

std::ostream& operator<<(std::ostream& oss, const GoL_Engine& gol_engine);