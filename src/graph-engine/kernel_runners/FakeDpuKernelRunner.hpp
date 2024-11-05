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
#include <assert.h>

#include "graph-engine/kernel_runner.hpp"

#include <xrt/xrt_device.h>
#include <experimental/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

namespace {
// Copy values from binary files into input buffer, expecting values are raw binary
void init_buf(vart::TensorBuffer *input, std::string &filename)
{
  uint64_t buff;
  size_t bytesize;
  std::tie(buff, bytesize) = input->data();

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);

  //ASSERT_TRUE(ifs.is_open()) << "Failed To Open File";

  ifs.read((char *)buff, bytesize);
}
}

/**
 * @class FakeDpuKernelRunner
 *
 * @brief
 * FakeDpuKernelRunner is a concrete KernelRunner class that implements a simple test kernel.
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
class FakeDpuKernelRunner : public DpuKernelRunner
{
public:
  // This enum defines the names of the xrt::kernel arguments, and creates a name -> position mapping
  enum
  {
    MODE_ARG,
    INPUT_ARG,
    PARAMETERS_ARG,
    OUTPUT_ARG,
    INTERMEDIATE_ARG,
    INSTRUCTIONS_ARG,
    INSTRUCTIONS_SIZE_ARG,
  };

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
  FakeDpuKernelRunner(Engine *engine, const xir::Subgraph *subgraph, xir::Attrs *attrs)
    : DpuKernelRunner(engine, subgraph, attrs)
  {
    if(!attrs || !attrs->has_attr("fake_dpu_binary_output_files"))
      throw std::runtime_error("FakeDpuKernelRunner cannot access metadata from attrs.");

    golden_ = attrs->get_attr<std::vector<std::string>>("fake_dpu_binary_output_files");
  }

  /**
   * Destroy FakeDpuKernelRunner object.
   */
  virtual ~FakeDpuKernelRunner() = default;

  /**
   * Run inference for this FakeDpuKernelRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   *
   * Concrete KernelRunner implementations must define this function.
   * FakeDpuKernelRunner will output golden data files for infrastructure testing.
   */
  virtual void run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) override
  {
    for(int i = 0; i < outputs.size(); i++)
      init_buf(outputs[i], golden_[i]);
  }

  /**
   * use xir::Attrs to pass info to RunnerExt at runtime.
   * return 0: good; 1: error, RunnerExt cannot find any useful info in Attrs
   */
  virtual int set_run_attrs(std::unique_ptr<xir::Attrs>&) override { return 0; }


protected:
    std::vector<std::string> golden_;
};
