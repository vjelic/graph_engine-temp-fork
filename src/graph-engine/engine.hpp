// Copyright 2021 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include "blockingconcurrentqueue.h"

static std::string get_env_var(const std::string& var,
                               const std::string& default_val = {}) {
#ifdef _WIN32
  char* value = nullptr;
  size_t size = 0;
  errno_t err = _dupenv_s(&value, &size, var.c_str());
  if ((!err && (value != nullptr)))
  {
      std::cout << "WARNING: Usage of environment variable XLNX_VART_FIRMWARE to provide xclbin file is going to be deprecated and will no longer be supported. A User can instead provide xclbin file location through attrs object. Please look at VitisAI/testcases/vairt/hello_world_no_env_var as a reference on how to provide xclbin file location. Detailed documentation is provided in the testcase repo." << std::endl;
  }
  std::string result =
      (!err && (value != nullptr)) ? std::string{value} : default_val;
  free(value);
#else
  const char* value = std::getenv(var.c_str());
  if (value != nullptr)
  {
      std::cout << "WARNING: Usage of environment variable XLNX_VART_FIRMWARE to provide xclbin file is going to be deprecated and will no longer be supported. A User can instead provide xclbin file location through attrs object. Please look at VitisAI/testcases/vairt/inference/test_graph_engine_no_env_var.cpp as a reference on how to provide xclbin file location. Detailed documentation is provided in the testcase repo." << std::endl;
  }
  std::string result = (value != nullptr) ? std::string{value} : default_val;
#endif
  return result;
}

namespace
{
  unsigned int constexpr MAX_WORKERS = 16;
  unsigned getNumWorkers()
  {
    return std::stoi(get_env_var("GRAPH_ENGINE_NUM_WORKERS", std::to_string(MAX_WORKERS)));
  }
}

/**
 * @class Engine
 *
 * @brief
 * An Engine object represents a task pool. Any function returning void can be submitted to this task pool, 
 * and a worker thread will pick it up and call it.
 *
 * @details
 * All GraphRunners and KernelRunners will have a pointer to this task pool, and can submit their
 * run() function to this task pool, as well as monitor its status.
 */
class Engine
{
  struct EngineConccurrentQueueTraits : public moodycamel::ConcurrentQueueDefaultTraits
  {
    static const int MAX_SEMA_SPINS = 100;
  };

public:
  enum TaskState
  {
    NEW,
    RUNNING,
    DONE
  };

  /**
   * Construct and initialize Engine object.
   *
   * Construct a queue of job_ids and statuses
   * this queue will be used to track in flight jobs
   * 
   * Launch a number of worker threads to wait for work 
   */
  Engine() : terminate_(false)
  {
    constexpr unsigned maxConcurrentTasks = 10000;
    const unsigned numWorkerThreads = getNumWorkers();

    // init task ids and task status
    task_status_.resize(maxConcurrentTasks, Engine::DONE);
    for (uint32_t i = 0; i < maxConcurrentTasks; i++)
      task_ids_.enqueue(i);

    // spawn threads
    for (unsigned i = 0; i < numWorkerThreads; i++)
    {
      threads_.emplace_back(std::thread([this]
                                        { run(); }));
      thread_worker_ids_[threads_.back().get_id()] = i;
    }
  }

  /**
   * Destroy Engine object.
   */
  ~Engine()
  {
    // collect threads
    terminate_ = true;
    for (unsigned i = 0; i < threads_.size(); i++)
      threads_[i].join();
  }

  /**
   * Submit a task to the task pool.
   *
   * @param task
   *  std::function object to be invoked by a worker.
   * @return
   *  Numeric identifier used to track the status of this task.
   */
  uint32_t enqueue(std::function<void()> task)
  {
    uint32_t id;
    task_ids_.wait_dequeue(id);               // block until we get a free ID from the pool
    task_status_[id] = Engine::NEW;           // mark task "new"
    taskq_.enqueue(std::make_pair(id, task)); // send task for thread to execute
    return id;
  }

  /**
   * Wait for specified job to complete.
   *
   * @param job_id
   *  The numerical job identifier to wait on.
   * @param timeout
   *  The timout in milliseconds to wait before aborting. A value of negative 1 will wait indefinitely.
   * @return
   *  Returns 0 on success.
   */
  void wait(uint32_t job_id, int timeout = -1)
  {
    // wait for task to be done
    const auto waitInterval = (timeout > 0) ? std::chrono::milliseconds(timeout) : std::chrono::seconds(5);
    const auto startTime = std::chrono::high_resolution_clock::now();

    {
      std::unique_lock<std::mutex> lock(status_mtx_);
      while (task_status_[job_id] != Engine::DONE)
      {
        (void)status_cvar_.wait_for(lock, waitInterval);

        if (timeout <= 0)
          continue; // no timeout requested, keep waiting

        auto currTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(
            currTime - startTime);

        if (elapsedTime.count() * 1000 > timeout)
          throw std::runtime_error("Error: task timeout: " + std::to_string(job_id));
      }
    }

    // return task id to id pool
    task_ids_.enqueue(job_id);
  }

  /**
   * Get the number of workers belonging to the task pool.
   *
   * @return
   * Number of workers in the task pool.
   */
  unsigned get_num_workers() const { return threads_.size(); }

  /**
   * Get the integral worker number from a thread identifier.
   *
   * @return
   * Worker index.
   * 
   * This function can be used by an executing worker to get a simple numerical self identifier.
   * This is helpful if you have global per worker resources in an array, and you want an index for that array.
   */
  unsigned get_worker_id(std::thread::id id)
  {
    auto it = thread_worker_ids_.find(id);
    if (it == thread_worker_ids_.end())
      throw std::runtime_error("Error: unknown worker thread id");
    return it->second;
  }

private:
  void run()
  {
    while (1)
    {
      // get new task from queue
      std::pair<int, std::function<void()>> task;
      bool found = taskq_.wait_dequeue_timed(task, std::chrono::milliseconds(5));
      if (!found)
      {
        // queue was empty
        if (terminate_)
          break;
        else
          continue;
      }

      // run task
      task_status_[task.first] = Engine::RUNNING;
      task.second();

      // report task done
      {
        task_status_[task.first] = Engine::DONE;
        std::unique_lock<std::mutex> lock(status_mtx_);
        status_cvar_.notify_all();
      }
    }
  }

  std::mutex status_mtx_;
  std::condition_variable status_cvar_;
  moodycamel::BlockingConcurrentQueue<std::pair<int, std::function<void()>>, EngineConccurrentQueueTraits> taskq_;
  moodycamel::BlockingConcurrentQueue<uint32_t, EngineConccurrentQueueTraits> task_ids_;
  std::vector<int> task_status_; // ID -> status
  std::atomic<bool> terminate_;
  std::vector<std::thread> threads_;
  std::unordered_map<std::thread::id, unsigned> thread_worker_ids_;
};
