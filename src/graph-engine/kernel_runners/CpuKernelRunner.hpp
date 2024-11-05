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

#include <iostream>
#include <memory>
#include <assert.h>

#include "graph-engine/kernel_runner.hpp"
#include "graph-engine/op_runner_factory.hpp"
#include "graph-engine/op_runner.hpp"

/**
 * @class CpuKernelRunner
 *
 * @brief
 * CpuKernelRunner is a concrete KernelRunner class that implements common kernel runner for all the subgraphs to be run on the CPU.
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
class CpuKernelRunner : public KernelRunner
{
public:
  /**
   * Constructor for concrete CpuKernelRunner.
   *
   * @param engine
   *  This is a pointer to a global task pool used for submitting execution requests.
   * @param subgraph
   *  This is a pointer to a subgraph. A subgraph is an object that aggregates
   *    the necessary meta data needed by a KernelRunner.
   * 
   * This is an implementation of a CPU Kernel that dispatches tasks to Op Runners to run ops on CPU.
   */
  CpuKernelRunner(Engine *engine, const xir::Subgraph *subgraph, xir::Attrs *attrs)
    : KernelRunner(engine, subgraph)
  {

    for(auto& op : subgraph->topological_sort())
    {
      if(op->get_type()=="pad-fix")
        this->op_runners_.emplace_back(OpRunnerFactory::create("PadOpRunner", engine_, op, attrs));
    }
  
    if(op_runners_.empty())
      throw std::runtime_error("Error: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " Could not determine OpRunner for any ops.");

    inputsOwned_ = create_inputs();
    outputsOwned_ = create_outputs();
    inputs_ = to_raw(inputsOwned_);
    outputs_ = to_raw(outputsOwned_);
  }

  /**
   * Destroy CpuKernelRunner object.
   */
  virtual ~CpuKernelRunner() = default;

  /**
   * Run inference for this CpuKernelRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   *
   * Concrete KernelRunner implementations must define this function.
   * CpuKernelRunner will call op runners with given inputs and outputs.
   */
  virtual void run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) override
  {
    
    for(int i=0; i < op_runners_.size(); i++)
    {
      op_runners_[i]->run(inputs, outputs);
    }
  
  }

  virtual std::vector<vart::TensorBuffer *> get_inputs() override { return inputs_; } 
  virtual std::vector<vart::TensorBuffer *> get_outputs() override { return outputs_; }

  /**
   * use xir::Attrs to pass info to RunnerExt at runtime.
   * return 0: good; 1: error, RunnerExt cannot find any useful info in Attrs
   */
  virtual int set_run_attrs(std::unique_ptr<xir::Attrs>&) override { return 0; }

protected:
  std::vector<std::unique_ptr<OpRunner>> op_runners_;

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputsOwned_;
  std::vector<std::unique_ptr<vart::TensorBuffer>> outputsOwned_;
  std::vector<vart::TensorBuffer*> inputs_;
  std::vector<vart::TensorBuffer*> outputs_;
 };
