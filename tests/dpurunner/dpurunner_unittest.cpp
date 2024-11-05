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

#include <vart/runner.hpp>

TEST(DpuRunnerTest, DpuRunnerTest)
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

  // Read in XMODEL
  std::unique_ptr<xir::Graph> graph = xir::Graph::deserialize("./the.xmodel");

  // Get the First FPGA Subgraph
  std::vector<xir::Subgraph *> subgraphs = graph->get_root_subgraph()->children_topological_sort();
  auto subgraph = *std::find_if(subgraphs.begin(), subgraphs.end(), [](xir::Subgraph *sg) {
    return sg->get_attr<std::string>("device") == "DPU";
  });
 
  auto dpuRunner = vart::Runner::create_runner_with_attrs(subgraph, attrs.get());

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs_owned = GraphEngine::create_inputs(dpuRunner);
  std::vector<std::unique_ptr<vart::TensorBuffer>> outputs_owned = GraphEngine::create_outputs(dpuRunner);
  
  // Convert to raw pointer view, for execute_async compatibility
  std::vector<vart::TensorBuffer*> inputs;
  std::transform(inputs_owned.cbegin(), inputs_owned.cend(), std::back_inserter(inputs),[](const std::unique_ptr<vart::TensorBuffer> &input){ return input.get(); });
  std::vector<vart::TensorBuffer*> outputs;
  std::transform(outputs_owned.cbegin(), outputs_owned.cend(), std::back_inserter(outputs),[](const std::unique_ptr<vart::TensorBuffer> &output){ return output.get(); });
  
  std::vector<std::string> goldenInputFiles = {"./golden_ifm.bin"};
  std::vector<std::string> goldenOutputFiles = {"./golden_ofm.bin"};

  for (unsigned i = 0; i < inputs.size(); i++)
  {
    init_buf(inputs[i], goldenInputFiles[i]);
  }

  auto job = dpuRunner->execute_async(inputs, outputs);

  dpuRunner->wait(job.first, -1);

  for (unsigned i = 0; i < outputs.size(); i++)
  {
    comp_buf(outputs[i], goldenOutputFiles[i]);
  }
}
