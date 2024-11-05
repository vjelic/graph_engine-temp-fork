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

TEST(MultiHwContextTest, MultiHwContextTest)
{
  
  // Higher Layer Takes Care Of Device/Xclbin/Context management
  xrt::device xrt_device(0);
  xrt::xclbin xrt_xclbin("./the.xclbin");
  xrt_device.register_xclbin(xrt_xclbin);
  xrt::hw_context xrt_hw_context(xrt_device, xrt_xclbin.get_uuid());
 
  xrt::hw_context xrt_hw_context1(xrt_device, xrt_xclbin.get_uuid());

  // Higher Layer Passes pointers to XRT objects via XIR Attrs 
  auto attrs = xir::Attrs::create();
  attrs->set_attr<xrt::device*>("xrt_device", &xrt_device);
  attrs->set_attr<xrt::xclbin*>("xrt_xclbin", &xrt_xclbin);
  attrs->set_attr<xrt::hw_context*>("xrt_hw_context", &xrt_hw_context);

  auto attrs1 = xir::Attrs::create();
  attrs1->set_attr<xrt::device*>("xrt_device", &xrt_device);
  attrs1->set_attr<xrt::xclbin*>("xrt_xclbin", &xrt_xclbin);
  attrs1->set_attr<xrt::hw_context*>("xrt_hw_context", &xrt_hw_context1);

  auto graphRunner = GraphEngine::create_graph_runner("./the.xmodel", attrs.get());
  auto graphRunner1 = GraphEngine::create_graph_runner("./the.xmodel", attrs1.get());

  auto inputs = graphRunner->get_inputs();
  auto outputs = graphRunner->get_outputs();

  auto inputs1 = graphRunner1->get_inputs();
  auto outputs1 = graphRunner1->get_outputs();

  std::vector<std::string> goldenInputFiles = {"./golden_ifm.bin"};
  std::vector<std::string> goldenOutputFiles = {"./golden_ofm.bin"};

  for (unsigned i = 0; i < inputs.size(); i++)
  {
    init_buf(inputs[i], goldenInputFiles[i]);
  }

  for (unsigned i = 0; i < inputs1.size(); i++)
  {
    init_buf(inputs1[i], goldenInputFiles[i]);
  }


  auto job = graphRunner->execute_async(inputs, outputs);
  auto job1 = graphRunner1->execute_async(inputs1, outputs1);

  graphRunner->wait(job.first, -1);
  graphRunner1->wait(job1.first, -1);

  for (unsigned i = 0; i < outputs.size(); i++)
  {
    comp_buf(outputs[i], goldenOutputFiles[i]);
  }

  for (unsigned i = 0; i < outputs1.size(); i++)
  {
    comp_buf(outputs1[i], goldenOutputFiles[i]);
  }

}
