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
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/graph/graph.hpp>
#include "graph-engine/engine.hpp"
#include "graph-engine/buffers.hpp"

/**
 * @class KernelRunner
 *
 * @brief
 * KernelRunner is an abstract base class used to define how a
 * concrete KernelRunner should be invoked.
 *
 * @details
 * All KernelRunners must be constructed with two arguments:
 *  - A pointer to a TaskPool "engine"
 *  - A pointer to an xir::subgraph
 * A KernelRunner must provide a run method that defines the transfer funtion
 * of how to process a batch of inputs and generate data to a batch of outputs.
 *
 * The base class also provides common functionality such as extracting input and output shapes.
 */
class KernelRunner : public vart::RunnerExt
{

public:
  /**
   * Construct Empty KernelRunner object.
   */
  KernelRunner() = delete;

  /**
   * Destroy KernelRunner object.
   */
  virtual ~KernelRunner() = default;

  /**
   * Constructor for base class to store common metadata.
   *
   * @param engine
   *  This is a pointer to a global task pool used for submitting execution requests.
   * @param subgraph
   *  This is a pointer to a subgraph. A subgraph is an object that aggregates
   *    the necessary meta data needed by a KernelRunner.
   */
  KernelRunner(Engine* engine, const xir::Subgraph* subgraph)
      : engine_(engine), subgraph_(subgraph)
  {
      auto tensors_in = subgraph_->get_sorted_input_tensors();
      auto tensors_out = subgraph_->get_sorted_output_tensors();
      for (auto tensor : tensors_in) {
          auto tensor_in = find_tensor(tensor, subgraph, true);
          if (!tensor_in->has_attr("ddr_addr"))
              throw std::runtime_error("No output ddr_addr exists for this tensor: " + tensor_in->get_name());
          if (!tensor_in->has_attr("stride"))
              throw std::runtime_error("No output stride exists for this tensor: " + tensor_in->get_name());
          auto dims = tensor_in->get_shape();
          auto tensor_t = xir::Tensor::create(tensor->get_name(), dims, tensor->get_data_type());
          auto attrs = tensor_in->get_attrs();
          tensor_t->set_attrs(std::move(attrs));
          intensors_.emplace_back(std::move(tensor_t));
      }
      for (auto tensor : tensors_out) {
          auto dims = tensor->get_shape();
          auto tensor_t = xir::Tensor::create(tensor->get_name(), dims, tensor->get_data_type());
          auto actual_output_tensor = find_tensor(tensor, subgraph, false);
          if (!actual_output_tensor->has_attr("ddr_addr"))
              throw std::runtime_error("No output ddr_addr exists for this tensor: " + actual_output_tensor->get_name());
          if (!actual_output_tensor->has_attr("stride"))
              throw std::runtime_error("No output stride exists for this tensor: " + actual_output_tensor->get_name());
          auto attrs = actual_output_tensor->get_attrs();
          tensor_t->set_attrs(std::move(attrs));
          outtensors_.emplace_back(std::move(tensor_t));
      }
      input_tensors_.reserve(intensors_.size());
      std::transform(intensors_.begin(), intensors_.end(), std::back_inserter(input_tensors_),
          [](auto& tensor) { return tensor.get(); });
      output_tensors_.reserve(outtensors_.size());
      std::transform(outtensors_.begin(), outtensors_.end(), std::back_inserter(output_tensors_),
          [](auto& tensor) { return tensor.get(); });
  }

  /**
   * Static function used by KernelRunnerFactory to construct concrete KernelRunners.
   *
   * @param engine
   *  This is a pointer to a global task pool used for submitting execution requests.
   * @param subgraph
   *  This is a pointer to a subgraph. A subgraph is an object that aggregates
   *    the necessary meta data needed by a KernelRunner.
   * @return
   *  Unique pointer to a concrete KernelRunner object.
   *
   * Essentially, this is just a method to call the constructor for the derived types,
   * and return a unique pointer to that object. By putting this in the base class we avoid
   * having to copy paste this code into every concrete KernelRunner.
   */
  template <typename T>
  static std::unique_ptr<KernelRunner> create(Engine *engine, const xir::Subgraph *subgraph, xir::Attrs *attrs)
  {
    return std::make_unique<T>(engine, subgraph, attrs);
  }

  /**
   * Run inference for this KernelRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   *
   * This is a pure virtual function which must be overridden by concrete KernelRunner implementations.
   */
  virtual void run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) = 0;

  /**
   * Submit this KernelRunner's run function to the task pool.
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
   * Submit this KernelRunners' run fuction to the task pool, and wait for job completion.
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
   * Get this KernelRunner's input_tensors_.
   *
   * @return
   *  Vector of xir::Tensor pointers from which the number of inputs and their shapes can be determined.
   */
  virtual std::vector<const xir::Tensor *> get_input_tensors() override
  {
    std::vector<const xir::Tensor*> tensors;
    std::transform(input_tensors_.cbegin(), input_tensors_.cend(), std::back_inserter(tensors),[](const xir::Tensor* tensor){return tensor;});
    return tensors;
  }

  /**
   * Get this KernelRunner's output_tensors_.
   *
   * @return
   *  Vector of xir::Tensor pointers from which the number of outputs and their shapes can be determined.
   */
  virtual std::vector<const xir::Tensor *> get_output_tensors() override
  {
    std::vector<const xir::Tensor*> tensors;
    std::transform(output_tensors_.cbegin(), output_tensors_.cend(), std::back_inserter(tensors),[](const xir::Tensor* tensor){return tensor;});
    return tensors;
  }

  /**
   * Construct and return TensorBuffers for this KernelRunner.
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
   * Construct and return TensorBuffers for this KernelRunner.
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
   * Get this KernelRunner's kernel name.
   *
   * @return
   *  This KernelRunner's name as std::string.
   */
  virtual std::string get_kernel_name()
  {
    // TODO: Error handling
    return subgraph_->get_attr<std::string>("kernel");
  }
  static std::vector<vart::TensorBuffer *>
  to_raw(std::vector<std::unique_ptr<vart::TensorBuffer>> &v) {
    std::vector<vart::TensorBuffer *> rv;
    std::transform(
        v.begin(), v.end(), std::back_inserter(rv),
        [](std::unique_ptr<vart::TensorBuffer> &sp) { return sp.get(); });
    return rv;
  }

  const xir::Tensor* find_tensor(const xir::Tensor* in_tensor, const xir::Subgraph* subgraph, bool isInput) {
      auto op_tmp = in_tensor->get_producer();
      auto out = op_tmp->get_output_tensor();
      if ((!isInput) && (op_tmp->get_type() == "download")) {
          auto input_ops = op_tmp->get_input_ops("input");
          out = input_ops[0]->get_output_tensor();
      }
      else if (!out->has_attr("reg_id")) {
          auto fanout_ops = op_tmp->get_fanout_ops();
          auto subgraph_ops = subgraph->get_ops();
          auto subgraph_ops1 = std::vector<const xir::Op*>(subgraph_ops.begin(), subgraph_ops.end());
          std::sort(fanout_ops.begin(), fanout_ops.end());
          std::sort(subgraph_ops1.begin(), subgraph_ops1.end());
          auto ops = std::vector<const xir::Op*>();
          std::set_intersection(fanout_ops.begin(), fanout_ops.end(),
              subgraph_ops1.begin(), subgraph_ops1.end(),
              std::back_inserter(ops));
          auto upload_op = ops.front();
          out = upload_op->get_output_tensor();

      }
      return out;

  }
  
protected:
  Engine *engine_;
  const xir::Subgraph *subgraph_;
  std::vector<xir::Tensor *> input_tensors_;
  std::vector<std::unique_ptr<xir::Tensor>> intensors_;
  std::vector<std::unique_ptr<xir::Tensor>> outtensors_;
  std::vector<xir::Tensor *> output_tensors_;
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputsOwned_;
  std::vector<std::unique_ptr<vart::TensorBuffer>> outputsOwned_;
  std::vector<vart::TensorBuffer *> inputs_;
  std::vector<vart::TensorBuffer *> outputs_;
};

using create_cb_t = std::function<std::unique_ptr<KernelRunner>(Engine *, const xir::Subgraph *, xir::Attrs *)>;
