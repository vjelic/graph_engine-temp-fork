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
#include "graph-engine/qos.hpp"

#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <experimental/xrt_hw_context.h>

TEST(Resnet50Test, Resnet50Test)
{
  
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

  GraphEngine::qos_type my_qos;
  my_qos["fps"] = 60;
  my_qos["ext-gops"] = 1100;
  attrs->set_attr<GraphEngine::qos_type*>("qos_params", &my_qos);

  auto graphRunner = GraphEngine::create_graph_runner("./the.xmodel", attrs.get());
  // auto graphRunner = GraphEngine::create_graph_runner("./the.xmodel");

  auto inputs = graphRunner->get_inputs();
  auto outputs = graphRunner->get_outputs();

  std::vector<std::string> goldenInputFiles = {"./golden_ifm.bin"};
  std::vector<std::string> goldenOutputFiles = {"./golden_ofm.bin"};

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
