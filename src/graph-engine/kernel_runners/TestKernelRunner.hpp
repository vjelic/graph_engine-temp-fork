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

/**
 * @class TestKernelRunner
 *
 * @brief
 * TestKernelRunner is a concrete KernelRunner class that implements a simple test kernel.
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
class TestKernelRunner : public KernelRunner
{
public:
  /**
   * Constructor for concrete KernelRunner.
   *
   * @param engine
   *  This is a pointer to a global task pool used for submitting execution requests.
   * @param subgraph
   *  This is a pointer to a subgraph. A subgraph is an object that aggregates
   *    the necessary meta data needed by a KernelRunner.
   * 
   * This is an example of how to define your own KernelRunner.
   * It is also used for unit testing.
   */
  TestKernelRunner(Engine *engine, const xir::Subgraph *subgraph, xir::Attrs *attrs)
    : KernelRunner(engine, subgraph) {}

  /**
   * Destroy TestKernelRunner object.
   */
  virtual ~TestKernelRunner() = default;

  /**
   * Run inference for this TestKernelRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   *
   * Concrete KernelRunner implementations must define this function.
   * TestKernelRunner will just take every input, add 1, and store the result at the output.
   */
  virtual void run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) override
  {
    std::cout << "TestKernelRunner run function was called!" << std::endl;

    // Set output = input + 1
    for (unsigned int i = 0; i < inputs.size(); i++)
    {
      uint64_t inData, outData;
      size_t inSize, outSize;
      std::tie(inData, inSize) = inputs[i]->data();
      std::tie(outData, outSize) = outputs[i]->data();
      assert(inSize == outSize);
      for (unsigned int j = 0; j < inSize; j++)
      {
        ((int8_t *)outData)[j] = ((int8_t *)inData)[j] + 1;
      }
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
  // Define your class members
  // TestKernelRunner doesn't need any
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputsOwned_;
  std::vector<std::unique_ptr<vart::TensorBuffer>> outputsOwned_;
  std::vector<vart::TensorBuffer*> inputs_;
  std::vector<vart::TensorBuffer*> outputs_;
};
