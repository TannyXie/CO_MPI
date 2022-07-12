#include <cstdio>

#include "co_mpi.hpp"

co_mpi::Task<void> sendtoself() {
  int number = 1;
  int recv = 2;
  co_await co_mpi::Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_SELF);
  co_await co_mpi::Recv(&recv,   1, MPI_INT, 0, 0, MPI_COMM_SELF);
  co_return;
}

co_mpi::Task<void> ping(int number, int taskid) {
  std::printf("I'm here in task 0 point 0\n");
  co_mpi::Depended d = {.id = {taskid}};
  co_await d;
  int steps = 10;
  for(int i=0; i<steps; ++i) {
    co_await co_mpi::Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    co_await co_mpi::Recv(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
  }
  std::printf("I'm here in task 0 point 1\n");
  co_return;
}

co_mpi::Task<void> pong(int number) {
  std::printf("I'm here in task 1 point 0\n");
  int steps = 10;
  for(int i=0; i<steps; ++i) {
    co_await co_mpi::Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    co_await co_mpi::Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  std::printf("I'm here in task 1 point 1\n");
  co_return;
}

int main(int argc, char* argv[]) {
  // NOTE: RAII implies every resource in a program should be tied to
  // the lifetime of an object. MPI itself is a resource. Destruction
  // of this object should call MPI_Finalize.
  auto env = co_mpi::Init(nullptr, nullptr);

  int rank = co_mpi::Comm_rank(MPI_COMM_WORLD);

  auto exe = co_mpi::single_thread_executor(env);
  if (rank == 0) {
    int taskid = exe.start(sendtoself());
    exe.start(ping(0, taskid));
  } else if (rank == 1) {
    exe.start(pong(1));
  }
  exe.Runtillfinish();
  co_mpi::Finalize();
}
