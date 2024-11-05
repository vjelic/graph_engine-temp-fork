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
#include <xrt/xrt_bo.h>

TEST(AksBuffersTest, AksBuffersTest)
{
  // Golden files for the test
  // Data is raw binary int8
  std::vector<std::string> goldenInputFiles = {"./golden_ifm.bin"};
  std::vector<std::string> goldenOutputFiles = {"./golden_ofm.bin"};

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

  // AKS Mode
  // I am thinking that AKS will want to bypass the CPU Nodes inside of the XMODEL
  // Because it will implement padding / softmax / etc
  attrs->set_attr<bool>("bypass_cpu", true);

  attrs->set_attr<bool>("fake_dpu", true);
  attrs->set_attr<std::vector<std::string>>("fake_dpu_binary_output_files", goldenOutputFiles);

  auto graphRunner = GraphEngine::create_graph_runner("./the.xmodel", attrs.get());

  auto input_tensors = graphRunner->get_input_tensors();
  auto output_tensors = graphRunner->get_output_tensors();

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs_owned;
  for (unsigned i = 0; i < input_tensors.size(); i++)
  {
    auto shape = input_tensors[i]->get_shape();
    EXPECT_TRUE( (std::vector<std::int32_t>{1,56,56,64} == shape) );

    auto dtype = input_tensors[i]->get_data_type();
    EXPECT_EQ(xir::DataType::XINT, dtype.type);

    auto dsize = input_tensors[i]->get_data_size(); // I think this is number of elements
    EXPECT_EQ(200704, dsize);

    auto stride = input_tensors[i]->get_attr<std::vector<std::int32_t>>("stride");
    EXPECT_TRUE( (std::vector<std::int32_t>{200704,3584,64,1} == stride) );

    auto buffer_type = input_tensors[i]->get_attr<std::int32_t>("buffer_type");
    EXPECT_EQ(2, buffer_type); // 2 means XRT SubBuffer

    auto ddr_addr = input_tensors[i]->get_attr<std::int32_t>("ddr_addr");
    EXPECT_EQ(0, ddr_addr);

    auto parent_bo = input_tensors[i]->get_attr<xrt::bo*>("parent_bo");
    EXPECT_TRUE(parent_bo != nullptr);

    inputs_owned.emplace_back(std::make_unique<XrtBuffer>(input_tensors[i], stride, *parent_bo, dsize, ddr_addr));
    EXPECT_TRUE(inputs_owned.back().get() != nullptr);
  }
  std::vector<std::unique_ptr<vart::TensorBuffer>> outputs_owned;
  for (unsigned i = 0; i < output_tensors.size(); i++)
  {
    auto shape = output_tensors[i]->get_shape();
    EXPECT_TRUE( (std::vector<std::int32_t>{1,56,56,64} == shape) );

    auto dtype = output_tensors[i]->get_data_type();
    EXPECT_EQ(xir::DataType::XINT, dtype.type);

    auto dsize = output_tensors[i]->get_data_size(); // I think this is number of elements
    EXPECT_EQ(200704, dsize);

    auto stride = output_tensors[i]->get_attr<std::vector<std::int32_t>>("stride");
    EXPECT_TRUE( (std::vector<std::int32_t>{200704,3584,64,1} == stride) );

    auto buffer_type = output_tensors[i]->get_attr<std::int32_t>("buffer_type");
    EXPECT_EQ(2, buffer_type); // 2 means XRT SubBuffer

    auto ddr_addr = output_tensors[i]->get_attr<std::int32_t>("ddr_addr");
    EXPECT_EQ(0, ddr_addr);

    auto parent_bo = output_tensors[i]->get_attr<xrt::bo*>("parent_bo");
    EXPECT_TRUE(parent_bo != nullptr);

    outputs_owned.emplace_back(std::make_unique<XrtBuffer>(output_tensors[i], stride, *parent_bo, dsize, ddr_addr));
    EXPECT_TRUE(outputs_owned.back().get() != nullptr);
  }

  // Convert to raw pointer view, for execute_async compatibility
  std::vector<vart::TensorBuffer*> inputs;
  std::transform(inputs_owned.cbegin(), inputs_owned.cend(), std::back_inserter(inputs),[](const std::unique_ptr<vart::TensorBuffer> &input){ return input.get(); });
  std::vector<vart::TensorBuffer*> outputs;
  std::transform(outputs_owned.cbegin(), outputs_owned.cend(), std::back_inserter(outputs),[](const std::unique_ptr<vart::TensorBuffer> &output){ return output.get(); });


  for (unsigned i = 0; i < inputs.size(); i++)
  {
    init_buf(inputs[i], goldenInputFiles[i]);
  }

  auto job = graphRunner->execute_async(inputs, outputs);

  graphRunner->wait(job.first, -1);

  for (unsigned i = 0; i < outputs.size(); i++)
  {
    comp_buf(outputs[i], goldenOutputFiles[i]);
  }

}
