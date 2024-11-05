#pragma once

#include <iostream>
#include <memory>
#include <assert.h>

#include "graph-engine/op_runner.hpp"

/**
 * @class PadOpRunner
 *
 * @brief
 * PadOpRunner is a concrete OpRunner class that implements zero padding in the CPU.
 *
 * @details
 * All OpRunners must be constructed with two arguments:
 *  - A pointer to a TaskPool "engine"
 *  - A pointer to an xir::op
 * An OpRunner must provide a run method that defines the transfer funtion
 * of how to process a batch of inputs and generate data to a batch of outputs.
 *
 * The base class also provides common functionality such as extracting input and output shapes.
 */
class PadOpRunner : public OpRunner
{
public:
  /**
   * Constructor for concrete OpRunner.
   *
   * @param engine
   *  This is a pointer to a global task pool used for submitting execution requests.
   * @param op
   *  This is a pointer to an xir::op. 
   * 
   * This is an implementation of a CPU Padding Op that zero pads a 4D tensor.
   * The input and output must be in NHWC order, and only constant zero padding is performed.
   */
  PadOpRunner(Engine *engine, const xir::Op *op, xir::Attrs *attrs)
    : OpRunner(engine, op)
  {
    // By definition pad-fix ops are single input, single output
    const auto& inputTensor = input_tensors_[0];
    const auto& outputTensor = output_tensors_[0];

    // Get io shapes
    inputShape_ = inputTensor->get_shape();
    outputShape_ = outputTensor->get_shape();

    // Get the padding parameters from the subgraph
    if(op->has_attr("paddings"))
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
  }

  /**
   * Destroy PadOpRunner object.
   */
  virtual ~PadOpRunner() = default;

  /**
   * Run inference for this PadOpRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   *
   * Concrete OpRunner implementations must define this function.
   * PadOpRunner will just take every input, add 1, and store the result at the output.
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

protected:
  std::vector<std::int32_t> inputShape_;
  std::vector<std::int32_t> outputShape_;
  std::vector<std::int32_t> paddings_;
  std::vector<std::pair<std::int32_t, std::int32_t>> dimPaddings_;
  std::vector<std::pair<std::int32_t, std::int32_t>> dstSlices_;
};
