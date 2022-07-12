#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "co_mpi.hpp"

#define M 8
#define N 8

#define X 2
#define Y 2

int bigmap[M][N];

int newm[M/X+2][N/Y+2];
int oldm[M/X+2][N/Y+2];

int *newmap[M/X+2];
int *oldmap[M/X+2];

enum directions {left, right, up, down};

MPI_Comm cart; 
int neigh[4];
int displacement[2];
int rank;

int block_size[2] = {M/X, N/Y}; 

// debug
void printnew(int);
void printold(int);
void printbig();



co_mpi::Task<void> initialize(){
// void initialize(){
  int dims[2] = {X, Y}, periods[2] = {1, 1}, coords[2];
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
  MPI_Cart_coords(cart, rank, 2, coords);
  MPI_Cart_shift(cart, 0, 1, &neigh[up], &neigh[down]);
  MPI_Cart_shift(cart, 1, 1, &neigh[left], &neigh[right]);

  displacement[0] = block_size[0] * coords[0];
  displacement[1] = block_size[1] * coords[1];

  // printf("rank %d coords are %d, %d, left %d, right %d, up %d, down %d\n", rank, coords[0], coords[1], neigh[left], neigh[right], neigh[up], neigh[down]);

  srand(2);
  if(rank == 0) {
    for(int i=0; i<M; ++i) {
      for(int j=0; j<N; ++j) {
        bigmap[i][j] = rand() % 100;
        bigmap[i][j] = bigmap[i][j] > 30 ? 0 : bigmap[i][j];
      }
    }

    printbig();


    for(int i = 1; i < 4; ++i) {
      co_yield co_mpi::Send( &bigmap, M*N, MPI_INT, i, 0, MPI_COMM_WORLD );
      // MPI_Ssend( &bigmap, M*N, MPI_INT, i, 0, MPI_COMM_WORLD );
    }
  } else {
    co_yield co_mpi::Recv( &bigmap, M*N, MPI_INT, 0, 0, MPI_COMM_WORLD );
    // MPI_Recv( &bigmap, M*N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
  }
  for(int i = 0; i < block_size[0]; ++i){
    for(int j = 0; j < block_size[1]; ++j) {
      newm[i+1][j+1] = oldm[i+1][j+1] = bigmap[i+displacement[0]][j+displacement[1]];
    }
  }
  for(int i=0; i<2+block_size[0];++i) {
    oldmap[i] = &oldm[i][0];
    newmap[i] = &newm[i][0];
  }

  co_return;

}

void swapmap(int* a[], int* b[]) {
  int* t;
  for(int i=0;i<block_size[0]+2;++i) {
    t = a[i];
    a[i] = b[i];
    b[i] = t;
  }
}

co_mpi::Task<void> sync() {
// void sync() {
  MPI_Datatype column;
  MPI_Request r[8];
  MPI_Type_vector(block_size[0], 1, 2+block_size[1], MPI_INT, &column);
  MPI_Type_commit(&column);
  if(rank % 2 == 0) {
    // MPI_Recv(&newmap[1][0],                1, column, neigh[left],   0, cart, MPI_STATUS_IGNORE);
    // MPI_Ssend(&newmap[1][block_size[1]],    1, column, neigh[right],  0, cart);
    // MPI_Recv(&newmap[1][block_size[1]+1],  1, column, neigh[right],  1, cart, MPI_STATUS_IGNORE);
    // MPI_Ssend(&newmap[1][1],                1, column, neigh[left],   1, cart);    

    co_yield co_mpi::Recv(&newmap[1][0],                1, column, neigh[left],   0, cart);
    co_yield co_mpi::Send(&newmap[1][block_size[1]],    1, column, neigh[right],  0, cart);
    co_yield co_mpi::Recv(&newmap[1][block_size[1]+1],  1, column, neigh[right],  1, cart);
    co_yield co_mpi::Send(&newmap[1][1],                1, column, neigh[left],   1, cart);

  }
  else {
    // MPI_Ssend(&newmap[1][block_size[1]],    1, column, neigh[right],  0, cart);
    // MPI_Recv(&newmap[1][0],                1, column, neigh[left],   0, cart, MPI_STATUS_IGNORE);
    // MPI_Ssend(&newmap[1][1],                1, column, neigh[left],   1, cart);
    // MPI_Recv(&newmap[1][block_size[1]+1],  1, column, neigh[right],  1, cart, MPI_STATUS_IGNORE);

    co_yield co_mpi::Send(&newmap[1][block_size[1]],    1, column, neigh[right],  0, cart);
    co_yield co_mpi::Recv(&newmap[1][0],                1, column, neigh[left],   0, cart);
    co_yield co_mpi::Send(&newmap[1][1],                1, column, neigh[left],   1, cart);
    co_yield co_mpi::Recv(&newmap[1][block_size[1]+1],  1, column, neigh[right],  1, cart);


  }

  if(rank / 2 == 0) {
    // MPI_Recv(&&newmap[0][1],                1, MPI_INT, neigh[up],    0, cart, MPI_STATUS_IGNORE);
    // MPI_Ssend(&newmap[block_size[0]][1],    block_size[1], MPI_INT, neigh[down],  0, cart);
    // MPI_Ssend(&newmap[1][1],                block_size[1], MPI_INT, neigh[up],    1, cart);
    // MPI_Recv(&newmap[block_size[0]+1][1],  block_size[1], MPI_INT, neigh[down],  1, cart, MPI_STATUS_IGNORE);
    co_yield co_mpi::Recv(&newmap[0][1],                block_size[1], MPI_INT, neigh[up],    0, cart);
    co_yield co_mpi::Send(&newmap[block_size[0]][1],    block_size[1], MPI_INT, neigh[down],  0, cart);
    co_yield co_mpi::Send(&newmap[1][1],                block_size[1], MPI_INT, neigh[up],    1, cart);
    co_yield co_mpi::Recv(&newmap[block_size[0]+1][1],  block_size[1], MPI_INT, neigh[down],  1, cart);
  } else {
    // MPI_Ssend(&newmap[block_size[0]][1],    1, MPI_INT, neigh[down],  0, cart);
    // MPI_Recv(&newmap[0][1],                block_size[1], MPI_INT, neigh[up],    0, cart, MPI_STATUS_IGNORE);
    // MPI_Recv(&newmap[block_size[0]+1][1],  block_size[1], MPI_INT, neigh[down],  1, cart, MPI_STATUS_IGNORE);
    // MPI_Ssend(&newmap[1][1],                block_size[1], MPI_INT, neigh[up],    1, cart);
    co_yield co_mpi::Send(&newmap[block_size[0]][1],    block_size[1], MPI_INT, neigh[down],  0, cart);
    co_yield co_mpi::Recv(&newmap[0][1],                block_size[1], MPI_INT, neigh[up],    0, cart);
    co_yield co_mpi::Recv(&newmap[block_size[0]+1][1],  block_size[1], MPI_INT, neigh[down],  1, cart);
    co_yield co_mpi::Send(&newmap[1][1],                block_size[1], MPI_INT, neigh[up],    1, cart);
  }

  // MPI_Irecv(&newmap[1][0],                1, column, neigh[left],   0, cart, &r[0]);
  // MPI_Irecv(&newmap[1][block_size[1]+1],  1, column, neigh[right],  1, cart, &r[1]);
  // MPI_Isend(&newmap[1][1],                1, column, neigh[left],   1, cart, &r[2]);
  // MPI_Isend(&newmap[1][block_size[1]],    1, column, neigh[right],  0, cart, &r[3]);
  // MPI_Irecv(&newmap[0][1],                block_size[1], MPI_INT, neigh[up],    0, cart, &r[4]);
  // MPI_Irecv(&newmap[block_size[0]+1][1],  block_size[1], MPI_INT, neigh[down],  1, cart, &r[5]);
  // MPI_Isend(&newmap[1][1],                block_size[1], MPI_INT, neigh[up],    1, cart, &r[6]);
  // MPI_Isend(&newmap[block_size[0]][1],    block_size[1], MPI_INT, neigh[down],  0, cart, &r[7]);
  // 
  //   MPI_Waitall(8, r, MPI_STATUSES_IGNORE);

  swapmap(newmap, oldmap);
  co_return;
}

int max(int i, int j) {
  return i>j ? i : j;
}

template<typename... Args>
int max(int i, Args... args) {
  return max(i, max(args...));
}

// co_mpi::Task<void> step() {
void step() {
  for(int i = 1; i < block_size[0]+2; ++i) {
    for(int j = 1; j < block_size[1]+2; ++j) {
      if(oldmap[i][j] != 0)
        newmap[i][j] = max(oldmap[i-1][j], oldmap[i+1][j], oldmap[i][j-1], oldmap[i][j+1], oldmap[i][j]);
    }
  }
  // printold(1);
}

void printnew(int full) {
  printf("rank %d, newmap:\n", rank);
  if(!full){
    for(int i=1;i<1+block_size[0]; ++i) {
      for(int j=1; j<1+block_size[1]; ++j) {
        printf("%5d ", newmap[i][j]);
      }
      printf("\n");
    }
  } else {
    for(int i=0;i<2+block_size[0]; ++i) {
      for(int j=0; j<2+block_size[1]; ++j) {
        printf("%5d ", newmap[i][j]);
      }
      printf("\n");
    }
  }
}

void printold(int full) {
  printf("rank %d, oldmap:\n", rank);
  if(!full){
    for(int i=1;i<1+block_size[0]; ++i) {
      for(int j=1; j<1+block_size[1]; ++j) {
        printf("%5d ", oldmap[i][j]);
      }
      printf("\n");
    }
  } else {
    for(int i=0;i<2+block_size[0]; ++i) {
      for(int j=0; j<2+block_size[1]; ++j) {
        printf("%5d ", oldmap[i][j]);
      }
      printf("\n");
    }
  }
}

void printbig() {
  printf("rank %d, bigmap:\n", rank);
  for(int i=0;i<M; ++i) {
    for(int j=0; j<N; ++j) {
      printf("%5d ", bigmap[i][j]);
    }
    printf("\n");
  }
}

int main(int argc, char** argv) {
  auto env = co_mpi::Init(nullptr, nullptr);

  rank = co_mpi::Comm_rank(MPI_COMM_WORLD);

  auto exe = co_mpi::single_thread_executor(env);

  exe.run(initialize());
  // initialize();
  exe.Runtillfinish();

  int maxstep = 10;
  if(argc > 1) 
    maxstep = atoi(argv[1]);
  

  for(int i=0; i < maxstep; ++i) {
    // exe.run(step());
    exe.run(sync());
    // sync();
    exe.Runtillfinish();
    step();
  }

  memset(bigmap, 0, M*N*sizeof(int));

  for(int i=1;i<block_size[0]+1; ++i) {
    for(int j=1; j<block_size[1]+1; ++j) {
      bigmap[displacement[0]+i-1][displacement[1]+j-1] = newmap[i][j];
    }
  }

  int result[M][N];
  memset(result, 0, M*N*sizeof(int));

  MPI_Reduce(bigmap, result, M*N, MPI_INT, MPI_SUM, 0, cart);
  if(rank == 0) {
    printf("result:\n");
    for(int i=0;i<M; ++i) {
      for(int j=0; j<N; ++j) {
        printf("%5d ", result[i][j]);
      }
      printf("\n");
    }
  }
  co_mpi::Finalize();
  return 0;



}