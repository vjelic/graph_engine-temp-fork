// Copyright 2022 Xilinx, Inc
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

#include <memory>
#include <stdexcept>
#include <vart/runner.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/graph/graph.hpp>
#include "graph-engine/engine.hpp"
#include "graph-engine/buffers.hpp"

/**
 * @class OpRunner
 *
 * @brief
 * OpRunner is an abstract base class used to define how a
 * concrete OpRunner should be invoked.
 *
 * @details
 * All OpRunners must be constructed with two arguments:
 *  - A pointer to a TaskPool "engine"
 *  - A pointer to an xir::Op
 * A OpRunner must provide a run method that defines the transfer funtion
 * of how to process a batch of inputs and generate data to a batch of outputs.
 *
 * The base class also provides common functionality such as extracting input and output shapes.
 */
class OpRunner : public vart::Runner
{

public:
  /**
   * Construct Empty OpRunner object.
   */
  OpRunner() = delete;

  /**
   * Destroy OpRunner object.
   */
  virtual ~OpRunner() = default;

  /**
   * Constructor for base class to store common metadata.
   *
   * @param engine
   *  This is a pointer to a global task pool used for submitting execution requests.
   * @param op
   *  This is a pointer to a op. A op is an object that aggregates
   *    the necessary meta data needed by a OpRunner.
   */
  OpRunner(Engine *engine, const xir::Op *op)
      : engine_(engine),
        op_(op),
        input_tensors_(op_->get_input_tensors()),
        output_tensors_({op_->get_output_tensor()}) {}

  /**
   * Static function used by OpRunnerFactory to construct concrete OpRunners.
   *
   * @param engine
   *  This is a pointer to a global task pool used for submitting execution requests.
   * @param op
   *  This is a pointer to a op. A op is an object that aggregates
   *    the necessary meta data needed by a OpRunner.
   * @return
   *  Unique pointer to a concrete OpRunner object.
   *
   * Essentially, this is just a method to call the constructor for the derived types,
   * and return a unique pointer to that object. By putting this in the base class we avoid
   * having to copy paste this code into every concrete OpRunner.
   */
  template <typename T>
  static std::unique_ptr<OpRunner> create(Engine *engine, const xir::Op *op, xir::Attrs *attrs)
  {
    return std::make_unique<T>(engine, op, attrs);
  }

  /**
   * Run inference for this OpRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   *
   * This is a pure virtual function which must be overridden by concrete OpRunner implementations.
   */
  virtual void run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) = 0;

  /**
   * Submit this OpRunner's run function to the task pool.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   * @return
   *  Pair of job_id and status. The job_id is used to reference this execution later in wait().
   *  Status of 0 indicates that the job was submittied successfully.
   */
  virtual std::pair<uint32_t, int> execute_async(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) override
  {
    auto job_id = engine_->enqueue([this, &inputs, &outputs]
                                   { this->run(inputs, outputs); });
    return std::pair<uint32_t, int>(job_id, 0);
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
  virtual int wait(int job_id, int timeout) override
  {
    engine_->wait(job_id, timeout);
    return 0;
  }

  /**
   * Submit this OpRunners' run fuction to the task pool, and wait for job completion.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   * @param timeout
   *  Optional. The timout in milliseconds to wait before aborting. A value of negative 1 will wait indefinitely.
   * @return
   *  Pair of job_id and status. The job_id is used to reference this execution later in wait().
   *  Status of 0 indicates that the job was submittied successfully.
   */
  virtual std::pair<uint32_t, int> execute(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs, int timeout = -1)
  {
    auto job = execute_async(inputs, outputs);
    job.second = wait(job.first, timeout);
    return job;
  }

  /**
   * Get this OpRunner's input_tensors_.
   *
   * @return
   *  Vector of xir::Tensor pointers from which the number of inputs and their shapes can be determined.
   */
  virtual std::vector<const xir::Tensor *> get_input_tensors() override
  {
    return input_tensors_;
  }

  /**
   * Get this OpRunner's output_tensors_.
   *
   * @return
   *  Vector of xir::Tensor pointers from which the number of outputs and their shapes can be determined.
   */
  virtual std::vector<const xir::Tensor *> get_output_tensors() override
  {
    return output_tensors_;
  }

  /**
   * Construct and return TensorBuffers for this OpRunner.
   *
   * @return
   *  Vector of unique_ptr of TensorBuffers.
   */
  virtual std::vector<std::unique_ptr<vart::TensorBuffer>> create_inputs()
  {
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs;
    std::transform(input_tensors_.cbegin(), input_tensors_.cend(), std::back_inserter(inputs),
                   [](const xir::Tensor *tensor)
                   { return std::make_unique<HostBuffer<int8_t>>(tensor, tensor->get_data_size()); });
    return inputs;
  }

  /**
   * Construct and return TensorBuffers for this OpRunner.
   *
   * @return
   *  Vector of unique_ptr of TensorBuffers.
   */
  virtual std::vector<std::unique_ptr<vart::TensorBuffer>> create_outputs()
  {
    std::vector<std::unique_ptr<vart::TensorBuffer>> outputs;
    std::transform(output_tensors_.cbegin(), output_tensors_.cend(), std::back_inserter(outputs),
                   [](const xir::Tensor *tensor)
                   { return std::make_unique<HostBuffer<int8_t>>(tensor, tensor->get_data_size()); });
    return outputs;
  }
  
  /**
   * Get this OpRunner's op name.
   *
   * @return
   *  This OpRunner's name as std::string.
   */
  virtual std::string get_op_name()
  {
    // TODO: Error handling
    return op_->get_name();
  }
  
protected:
  Engine *engine_;
  const xir::Op *op_;
  std::vector<const xir::Tensor *> input_tensors_;
  std::vector<const xir::Tensor *> output_tensors_;
};

using create_cb_op_t = std::function<std::unique_ptr<OpRunner>(Engine *, const xir::Op *, xir::Attrs *)>;
