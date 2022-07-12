#include <coroutine>
#include <iostream>
#include <concepts>
#include <vector>
#include <map>
#include <mpi.h>
int debug = 0;
// int rank;
namespace co_mpi
{
  struct Depended {
    std::vector<int> id;

    constexpr bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) {}
    constexpr void await_resume() const noexcept {}
  };

  struct Request {
    std::coroutine_handle<> h_; // same handle as Task and promise of Task
    MPI_Request r;
    int task_id;


    // constexpr bool await_ready() const noexcept {
    bool await_ready() noexcept {
      int flag = 0;
      MPI_Request re = r;
      MPI_Test(&re, &flag, MPI_STATUS_IGNORE);
      r = re;
      if(flag) return true;
      else return false;
    }
    void await_suspend(std::coroutine_handle<> h) {}
    constexpr void await_resume() const noexcept {}


  };

  template<typename T>
  class Task {
    public:
    struct promise_type {
      MPI_Request request_ = MPI_REQUEST_NULL;
      int         status; // 0 for ready, 1 for waiting for communication, 2 for waiting for dependency, 3 for completed
      std::vector<int> depend;
      


      Task get_return_object() {
        return {
          .h_ = std::coroutine_handle<promise_type>::from_promise(*this)
        };
      }
      //TODO: register the dependency
      std::suspend_never initial_suspend() { return {}; }
      std::suspend_always final_suspend() noexcept { return {}; }
      void unhandled_exception() {}
      std::suspend_always yield_value(MPI_Request request) {
        request_ = request;
        status = 1;
        return {};
      }
      Request await_transform(MPI_Request req) {
        request_ = req;
        status = 1;
        return {.r = req};
      }
      std::suspend_always await_transform(Depended d) {
        depend = d.id;

        status = 0;
        return {};
      }
      void return_void() { status = 3; }
    };

    MPI_Request request = MPI_REQUEST_NULL;
    int         status;
    std::vector<int> deps;


    std::coroutine_handle<promise_type> h_;
    operator std::coroutine_handle<promise_type>() const { return h_; }
    operator std::coroutine_handle<>() const { return h_; }

    bool finished() { return status == 3;}
  };


  // comm is an Awaitable
  struct Environment {
  public:
    int rank;
  };

  int Comm_rank(MPI_Comm m) {
    int rank;
    MPI_Comm_rank( m, &rank);
    return rank;
  }

  struct Executor {
    Environment env_;

    std::vector< Task<void> >   tasks;
    std::vector< MPI_Request >  requests;
    std::vector< int >          request_index;
    
    // to avoid deep stack, schedule should be coroutine
    // schedule only returns when all the tasks complete executing
    void schedule() {
      int count = 0;
      while(1) {
        int i;
        int flag = 0;
        for(i=0; i<tasks.size(); ++i) {
          switch(tasks[i].status) {
            case 3: break;
            case 0: continuetask(i); // continue task
                    flag = 1;
                    break;
            case 1: break;
            case 2: for(std::vector<int>::iterator it = tasks[i].deps.begin(); it!=tasks[i].deps.end(); ++it) {
                      if(tasks[*it].status == 3) tasks[i].deps.erase(it);
                    }
                    if(tasks[i].deps.size() == 0) {
                      tasks[i].status = 0;
                      continuetask(i);
                      flag = 1;
                    }
                    break;
          }
        }
        if(flag) continue;
        else if(requests.size() == 0) break;
        else {
          // wait for a task and set ready
          MPI_Waitany(requests.size(), &requests[0], &i, MPI_STATUS_IGNORE);
          requests.erase(requests.begin()+i);
          int taskid = request_index[i];
          request_index.erase(request_index.begin()+i);

          tasks[taskid].status = 0;
          tasks[taskid].request = MPI_REQUEST_NULL;
        }
      }
    }

    template<typename T>
    int start(Task<T> ro) {
      std::coroutine_handle<Task<void>::promise_type> handle = ro.h_;
      {
        // expand promise to Task
        ro.status = handle.promise().status;
        ro.request = handle.promise().request_;
        ro.deps = handle.promise().depend;
      }
      switch(handle.promise().status) {
        case 3: break;
        case 1: requests.push_back(ro.request);
                request_index.push_back(tasks.size());
                break;
        case 2: // test on the completion of the depended task, if completed, set ready
                for(std::vector<int>::iterator it = ro.deps.begin(); it!=ro.deps.end(); ++it) {
                  if(tasks[*it].status == 3) ro.deps.erase(it);
                }
                if(ro.deps.size() == 0) ro.status = 0;
                break;
        default:ro.status = 0;
      }
      tasks.push_back(ro);
      return tasks.size()-1;

    }

    Task<void> continuetask(int i) { // continue the ith task
      Task<void> &t = tasks[i];
      t.h_();
      // unfold the values from the promise type
      t.status = t.h_.promise().status;
      t.request = t.h_.promise().request_;
      if(t.status == 1) {
        requests.push_back(t.request);
        request_index.push_back(i);
      }

      return t;
    }

    void Runtillfinish() {
      schedule();
    }

  };

  Environment Init(int* argc, char*** argv){
    MPI_Init(argc, argv);
    return { 0 };
  }
  
  Executor single_thread_executor(Environment env) {
    Executor e = { env };
    return e;
  }

  void Finalize() {
    MPI_Finalize();
  }

  MPI_Request Recv(void* buffer, int count, MPI_Datatype type, int src, int tag, MPI_Comm comm) {
    // change to non-blocking and return it immediately
    MPI_Request request;
    MPI_Irecv( buffer, count, type, src, tag, comm, &request );
    return request;
  }

  MPI_Request Send(void* buffer, int count, MPI_Datatype type, int des, int tag, MPI_Comm comm) {
    MPI_Request request;
    MPI_Isend( buffer, count, type, des, tag, comm, &request );
    return request;
  }
}