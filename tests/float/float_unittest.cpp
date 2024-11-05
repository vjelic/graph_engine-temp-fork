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

#include <gtest/gtest.h>

#include "buffer_io.hpp"
#include "create_test_graph.hpp"
#include "graph-engine/graph-engine.hpp"

#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <experimental/xrt_hw_context.h>

TEST(FloatTest, FloatTest)
{
  std::vector<std::string> goldenFloatInputFiles = {"./golden_ifm_float.bin"};
  std::vector<std::string> goldenIntInputFiles = {"./golden_ifm.bin"};
  std::vector<std::string> goldenIntOutputFiles = {"./golden_ofm.bin"};
  std::vector<std::string> goldenFloatOutputFiles = {"./golden_ofm_float.bin"};

  // Higher Layer Takes Care Of Device/Xclbin/Context management
  xrt::device xrt_device(0);
  xrt::xclbin xrt_xclbin("./the.xclbin");
  xrt_device.register_xclbin(xrt_xclbin);
  xrt::hw_context xrt_hw_context(xrt_device, xrt_xclbin.get_uuid());

  // Higher Layer Passes pointers to XRT objects via XIR Attrs 
  auto attrs = xir::Attrs::create();
  attrs->set_attr<xrt::device*>("xrt_device", &xrt_device);
  attrs->set_attr<xrt::xclbin*>("xrt_xclbin", &xrt_xclbin);
  attrs->set_attr<xrt::hw_context*>("xrt_hw_context", &xrt_hw_context);
  
  auto graphRunner = GraphEngine::create_graph_runner("./the.xmodel", attrs.get());

  auto input_scale_factors = GraphEngine::get_input_scale_factors(graphRunner);

  EXPECT_EQ(std::vector<float>{0.5f}, input_scale_factors);
  
  auto output_scale_factors = GraphEngine::get_output_scale_factors(graphRunner);
  
  EXPECT_EQ(std::vector<float>{0.25f}, output_scale_factors);

  auto inputs = graphRunner->get_inputs();
  
  auto inputTensors = graphRunner->get_input_tensors();

  auto outputs = graphRunner->get_outputs();
  
  auto outputTensors = graphRunner->get_output_tensors();
  
  // Allocate The Buffers for holding the floating point data
  // This could be to capture data coming from OpenCV  
  std::vector<std::vector<float>> floatInputs;
  for (unsigned i = 0; i < inputs.size(); i++)
  {
    floatInputs.push_back(std::vector<float>(inputTensors[i]->get_data_size()));
  }
  
  std::vector<std::vector<float>> floatOutputs;
  for (unsigned i = 0; i < outputs.size(); i++)
  {
    floatOutputs.push_back(std::vector<float>(outputTensors[i]->get_data_size()));
  }

  // Fill float buffers with data
  for (unsigned i = 0; i < inputs.size(); i++)
  {
    init_buf(floatInputs[i].data(), sizeof(float)*floatInputs[i].size(), goldenFloatInputFiles[i]);
  }
  
  // Fill int buffers with data
  for (unsigned i = 0; i < outputs.size(); i++)
  {
    init_buf(outputs[i], goldenIntOutputFiles[i]);
  }
  // Float -> Fix w/ Scale
  for (unsigned i = 0; i < inputs.size(); i++)
  {
    GraphEngine::copy_buffer(inputs[i], floatInputs[i]);
  }
  
  // Fix -> Float w/ Scale
  for (unsigned i = 0; i < outputs.size(); i++)
  {
    GraphEngine::copy_buffer(floatOutputs[i], outputs[i]);
  }

  for (unsigned i = 0; i < inputs.size(); i++)
  {
    comp_buf(inputs[i], goldenIntInputFiles[i]);
  }
  
  for (unsigned i = 0; i < outputs.size(); i++)
  {
    comp_buf(floatOutputs[i].data(), sizeof(float)*floatOutputs[i].size(), goldenFloatOutputFiles[i]);
  }
}
