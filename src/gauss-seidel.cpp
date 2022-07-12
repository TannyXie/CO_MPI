#include <mpi.h>
#include <algorithm>
#include <sys/time.h>
#include <string>
#include <cmath>
#include <iostream>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <fstream>
#include <getopt.h>

#define BSX 1024
#define BSY 1024

#ifdef TAMPI
int *serial = nullptr;
#else
int *serial = (int *) 1;
#endif


struct config {
  double *matrix;
  int timesteps;
};

inline void solveBlock(double *matrix, int nbx, int nby, int bx, int by)
{
	double &targetBlock = matrix[bx*nby + by];
	const double &centerBlock = matrix[bx*nby + by];
	const double &topBlock    = matrix[(bx-1)*nby + by];
	const double &leftBlock   = matrix[bx*nby + (by-1)];
	const double &rightBlock  = matrix[bx*nby + (by+1)];
	const double &bottomBlock = matrix[(bx+1)*nby + by];
	
	double sum = 0.0;
	for (int x = 0; x < BSX; ++x) {
		const row_t &topRow = (x > 0) ? centerBlock[x-1] : topBlock[BSX-1];
		const row_t &bottomRow = (x < BSX-1) ? centerBlock[x+1] : bottomBlock[0];
		
		for (int y = 0; y < BSY; ++y) {
			double left = (y > 0) ? centerBlock[x][y-1] : leftBlock[x][BSY-1];
			double right = (y < BSY-1) ? centerBlock[x][y+1] : rightBlock[x][0];
			
			double value = 0.25 * (topRow[y] + bottomRow[y] + left + right);
			double diff = value - targetBlock[x][y];
			sum += diff * diff;
			targetBlock[x][y] = value;
		}
	}
}

// inline void sendFirstComputeRow(block_t *matrix, int nbx, int nby, int rank, int rank_size)
void sendFirstComputeRow(double *matrix, int nbx, int nby, int rank, int rank_size)
{
	for (int by = 1; by < nby-1; ++by) {
		// #pragma oss task label(send first compute row) in(([nbx][nby]matrix)[1][by]) inout(*serial)
		MPI_Send(&matrix[nby+by][0], BSY, MPI_DOUBLE, rank - 1, by, MPI_COMM_WORLD);
	}
}

// inline void sendLastComputeRow(block_t *matrix, int nbx, int nby, int rank, int rank_size)
void sendLastComputeRow(double *matrix, int nbx, int nby, int rank, int rank_size)
{
	for (int by = 1; by < nby-1; ++by) {
		// #pragma oss task label(send last compute row) in(([nbx][nby]matrix)[nbx-2][by]) inout(*serial)
		MPI_Send(&matrix[(nbx-2)*nby + by][BSX-1], BSY, MPI_DOUBLE, rank + 1, by, MPI_COMM_WORLD);
	}
}

inline void receiveUpperBorder(double *matrix, int nbx, int nby, int rank, int rank_size)
{
	for (int by = 1; by < nby-1; ++by) {
		// #pragma oss task label(receive upper border) out(([nbx][nby]matrix)[0][by]) inout(*serial)
		MPI_Recv(&matrix[by][BSX-1], BSY, MPI_DOUBLE, rank - 1, by, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

inline void receiveLowerBorder(double *matrix, int nbx, int nby, int rank, int rank_size)
{
	for (int by = 1; by < nby-1; ++by) {
		// #pragma oss task label(receive lower border) out(([nbx][nby]matrix)[nbx-1][by]) inout(*serial)
		MPI_Recv(&matrix[(nbx-1)*nby + by][0], BSY, MPI_DOUBLE, rank + 1, by, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

inline void solveGaussSeidel(double *matrix, int nbx, int nby, int rank, int rank_size)
{
	if (rank != 0) {
		sendFirstComputeRow(matrix, nbx, nby, rank, rank_size);
		receiveUpperBorder(matrix, nbx, nby, rank, rank_size);
	}
	
	if (rank != rank_size - 1) {
		receiveLowerBorder(matrix, nbx, nby, rank, rank_size);
	}
	
	for (int bx = 1; bx < nbx-1; ++bx) {
		for (int by = 1; by < nby-1; ++by) {
			// #pragma oss task label(gauss seidel)     \
			// 		in(([nbx][nby]matrix)[bx-1][by]) \
			// 		in(([nbx][nby]matrix)[bx][by-1]) \
			// 		in(([nbx][nby]matrix)[bx][by+1]) \
			// 		in(([nbx][nby]matrix)[bx+1][by]) \
			// 		inout(([nbx][nby]matrix)[bx][by])
			solveBlock(matrix, nbx, nby, bx, by);
		}
	}
	
	if (rank != rank_size - 1) {
		sendLastComputeRow(matrix, nbx, nby, rank, rank_size);
	}
}

double solve(struct config &conf, int rowBlocks, int colBlocks)
{
	int rank, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	
	double *matrix = conf.matrix;
	const int timesteps = conf.timesteps;
	
	for (int t = 0; t < timesteps; ++t) {
		solveGaussSeidel(matrix, rowBlocks, colBlocks, rank, rank_size);
	}
	
	// #pragma oss taskwait
	MPI_Barrier(MPI_COMM_WORLD);
	
	return IGNORE_RESIDUAL;
}


int main(int argc, char **argv)
{
  MPI_Init(NULL, NULL);
  int rank;
  int rank_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

  struct config conf;
  conf.timesteps = 10;
  int colBlocks = 10;
  int rowBlocksPerRank = 2;
  conf.matrix = (double*) malloc(20*sizeof(double));
	double start = getTime();
	solve(conf, rowBlocksPerRank, colBlocks);
	double end = getTime();
	
	if (!rank) {
		long totalElements = (long)rowBlocksPerRank * (long)colBlocks;
		double performance = totalElements * (long)conf.timesteps;
		performance = performance / (end - start);
		performance = performance / 1000000.0;
		
		int threads = 1;
		
		fprintf(stdout, "rows, %d, cols, %d, rows_per_rank, %d, total, %ld, total_per_rank, %ld, bs, %d,"
				" ranks, %d, threads, %d, timesteps, %d, time, %f, performance, %f\n",
				rowBlocksPerRank, colBlocks, rowBlocksPerRank / rank_size, totalElements, totalElements / rank_size,
				BSX, rank_size, threads, conf.timesteps, end - start, performance);
	}
		
	MPI_Finalize();
	
	return 0;
}