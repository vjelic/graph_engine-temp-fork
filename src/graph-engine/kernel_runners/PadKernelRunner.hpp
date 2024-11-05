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
 * @class PadKernelRunner
 *
 * @brief
 * PadKernelRunner is a concrete KernelRunner class that implements zero padding in the CPU.
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
class PadKernelRunner : public KernelRunner
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
   * This is an implementation of a CPU Padding Kernel that zero pads a 4D tensor.
   * The input and output must be in NHWC order, and only constant zero padding is performed.
   */
  PadKernelRunner(Engine *engine, const xir::Subgraph *subgraph, xir::Attrs *attrs)
    : KernelRunner(engine, subgraph)
  {
    // By definition pad-fix ops are single input, single output
    const auto& inputTensor = input_tensors_[0];
    const auto& outputTensor = output_tensors_[0];

    // Get io shapes
    inputShape_ = inputTensor->get_shape();
    outputShape_ = outputTensor->get_shape();

    // Get the padding parameters from the subgraph
    for(auto& op : subgraph->get_ops())
      if(op->get_type()=="pad-fix" && op->has_attr("paddings"))
        paddings_ = op->get_attr<std::vector<std::int32_t>>("paddings");

    // Every dimension will have two parameters... before, after
    // Convert this into a vector of pairs, so we can iterate over dimensions easier
    for(unsigned int i = 0; i < paddings_.size(); i += 2)
      dimPaddings_.emplace_back(paddings_[i], paddings_[i+1]);

    // Compute index ranges in destination where data will need to be copied
    for(unsigned int i = 0; i < inputShape_.size(); i++)
    {
      dstSlices_.emplace_back(dimPaddings_[i].first, dimPaddings_[i].first + inputShape_[i]);
    }
    inputsOwned_ = create_inputs();
    outputsOwned_ = create_outputs();
    inputs_ = to_raw(inputsOwned_);
    outputs_ = to_raw(outputsOwned_);
  }

  /**
   * Destroy PadKernelRunner object.
   */
  virtual ~PadKernelRunner() = default;

  /**
   * Run inference for this PadKernelRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   *
   * Concrete KernelRunner implementations must define this function.
   * PadKernelRunner will just take every input, add 1, and store the result at the output.
   */
  virtual void run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) override
  {
    
    // By definition pad-fix ops are single input, single output
    auto &input = inputs[0];
    auto &output = outputs[0];

    // Get the raw pointers, and the size in bytes of io buffers
    std::uint64_t inputRaw, outputRaw;
    size_t inputSize, outputSize; 
    std::tie(inputRaw, inputSize) = input->data();
    std::tie(outputRaw, outputSize) = output->data();

    const auto *inputPtr = reinterpret_cast<std::int8_t*>(inputRaw); 
    auto *outputPtr = reinterpret_cast<std::int8_t*>(outputRaw);

    unsigned int srcIdx = 0;
    unsigned int dstIdx = 0;

    // Iterate across every possible output element
    // If the indexes do not correlate to an element in the input
    //   then zeroize 
    // Else copy the value from the input array
    for (unsigned int on = 0; on < outputShape_[0]; on++)
    {
      for (unsigned int oh = 0; oh < outputShape_[1]; oh++)
      {
        for (unsigned int ow = 0; ow < outputShape_[2]; ow++)
        {
          for (unsigned int oc = 0; oc < outputShape_[3]; oc++)
          {
            if (
              on < dstSlices_[0].first   || 
              on >= dstSlices_[0].second ||
              oh < dstSlices_[1].first   || 
              oh >= dstSlices_[1].second || 
              ow < dstSlices_[2].first   || 
              ow >= dstSlices_[2].second || 
              oc < dstSlices_[3].first   || 
              oc >= dstSlices_[3].second
            ) 
              outputPtr[dstIdx++] = 0;
            else
              outputPtr[dstIdx++] = inputPtr[srcIdx++];
          }
        }
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
  std::vector<std::int32_t> inputShape_;
  std::vector<std::int32_t> outputShape_;
  std::vector<std::int32_t> paddings_;
  std::vector<std::pair<std::int32_t, std::int32_t>> dimPaddings_;
  std::vector<std::pair<std::int32_t, std::int32_t>> dstSlices_;

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputsOwned_;
  std::vector<std::unique_ptr<vart::TensorBuffer>> outputsOwned_;
  std::vector<vart::TensorBuffer*> inputs_;
  std::vector<vart::TensorBuffer*> outputs_;
};
