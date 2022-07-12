#include <iostream>
#include <coroutine>
#include <concepts>
#include "co_mpi.hpp"

static co_mpi::Comm comm;

// this is uncomfortable that you have to pass the communicator into the task
co_mpi::ReturnObject task0( int number );
co_mpi::ReturnObject task1( int number );

int main() {
  comm = co_mpi::Init(NULL,NULL);

  int rank, size;
  rank = comm.rank();
  // size = comm.size();

  int number0, number1;
  
  if(rank == 0)
    // task0(number0);
    start( task0, number0 ); 
  else 
    // task1(number1);
    start( task1, number1 );
  
  comm.finalize();
}

co_mpi::ReturnObject task0( int number ) {
  int task_number = comm.register_coroutine();
  printf("I'm here in task 0 point 0\n");
  co_await Send( &number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD );
  co_await Recv( &number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD );
  printf("I'm here in task 0 point 1\n");
}

co_mpi::ReturnObject task1( int number ) {
  int task_number = comm.register_coroutine(); // the task number is used to identify the 
  printf("I'm here in task 1 point 0\n");
  co_await comm.Recv( &number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, task_number );
  co_await comm.Send( &number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, task_number );
  printf("I'm here in task 1 point 1\n");
}