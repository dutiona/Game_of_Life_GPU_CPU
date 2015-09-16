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
	void run_serial();
	void run_omp(size_t nb_thread);
	void run_std_thread(size_t nb_thread);

	bool allAlive() const;
	bool allDead() const;
	const Grid& getGrid() const;

private:

	const Grid* do_step();
	const Grid* do_step_omp();
	const Grid* do_step_std_thread(size_t nb_thread);

	void kill_lowNeighbours();
	void live_2to3neighbours();
	void kill_highNeighbours();
	void resurect_highNeighbours();

	void notifyObservers();

	Grid grid_;
	size_t step_number_;
	size_t max_step_;

	const size_t fill_thresold_; //% de la grille initialisée à Alive

	std::vector<std::shared_ptr<const IObserver>> observers_;
};

std::ostream& operator<<(std::ostream& oss, const GoL_Engine& gol_engine);