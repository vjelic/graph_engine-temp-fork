/*
 * Copyright 2022 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <vector>
#include <string>
#include "xir/graph/graph.hpp"

xir::Subgraph *create_test_graph(int z)
{

  // This information should come from a meta-data file, or command line, or other method
  std::vector<std::int32_t> inShape = {1, 224, 224, 3};
  std::vector<std::int32_t> outShape = {1, 224, 224, 3};
  std::string deviceName = "CPU";              // Call it what you want, this string is used by application to pick the subgraph to run
  std::string kernelName = "TestKernelRunner"; // Call it what you want, to pick the host-side controller code to run
  std::vector<char> mc_code = {'T', 'h', 'e', 'I', 'n', 's', 't', 'r', 'u', 'c', 't', 'i', 'o', 'n', 's'};
  std::vector<char> params = {'T', 'h', 'e', 'P', 'a', 'r', 'a', 'm', 'e', 't', 'e', 'r', 's'};

  // Create a graph object
  std::unique_ptr<xir::Graph> graph = xir::Graph::create("graph_test");

  // Add a "data-fix" operator to simulate an input
  auto data_input_attrs = xir::Attrs::create();
  data_input_attrs->set_attr<std::vector<std::int32_t>>("shape", inShape);
  data_input_attrs->set_attr<std::string>("data_type", "XINT8");
  auto data_input = graph->add_op("data-input",
                                  "data-fix",
                                  std::move(data_input_attrs),
                                  {});

  // Add custom operator so we can define our own output shape
  //   ignoring shape inference
  auto custom_attrs = xir::Attrs::create();
  custom_attrs->set_attr<std::vector<std::int32_t>>("shape", outShape);
  custom_attrs->set_attr<std::string>("data_type", "XINT8");
  auto custom_op = graph->add_op("data-output",
                                 "xmagic",
                                 std::move(custom_attrs),
                                 {{"input", {data_input}}});

  auto *root = graph->get_root_subgraph();

  root->set_attr<std::map<std::string, std::string>>(
        "runner",
        {{"ref", "libgraph-engine.so"}, {"sim", "libgraph-engine.so"}, {"run", "libgraph-engine.so"}});

  // Every operator is now part of its own subgraph
  root->create_children();

  // Merge a set of operators into a new subgraph
  auto *input = root->merge_children({root->find_op("data-input")});
  auto *test = root->merge_children({root->find_op("data-output")});

  // Set the important graph meta-data that the runtime will need to read
  test->set_name("test_subgraph");
  test->set_attr<std::string>("device", deviceName);
  test->set_attr<std::string>("kernel", kernelName);
  test->set_attr<std::vector<char>>("mc_code", mc_code);
  test->set_attr<std::vector<char>>("params", params);

#if 0
  // Sanity check graph meta-data
  for (auto &subgraph : root->children_topological_sort())
  {
    std::cout << subgraph->get_name() << std::endl;

    if (subgraph->has_attr("device"))
      std::cout << "Device attribute: " << subgraph->get_attr<std::string>("device") << std::endl;
    if (subgraph->has_attr("kernel"))
      std::cout << "Kernel attribute: " << subgraph->get_attr<std::string>("kernel") << std::endl;
    if (subgraph->has_attr("mc_code"))
      std::cout << "This Subgraph has binary instructions" << std::endl;
    if (subgraph->has_attr("params"))
      std::cout << "This Subgraph has binary parameters / weights/ biases" << std::endl;

    auto inputTensors = subgraph->get_input_tensors();

    std::cout << "Inputs:" << std::endl;

    for (auto &inputTensor : inputTensors)
    {
      std::cout << inputTensor->get_name() << std::endl;
      auto shape = inputTensor->get_shape();
      for (auto extent : shape)
      {
        std::cout << "  " << extent << std::endl;
      }
    }

    auto outputTensors = subgraph->get_output_tensors();

    std::cout << "Outputs:" << std::endl;

    for (auto &outputTensor : outputTensors)
    {
      std::cout << outputTensor->get_name() << std::endl;
      auto shape = outputTensor->get_shape();
      for (auto extent : shape)
      {
        std::cout << "  " << extent << std::endl;
      }
    }
  }
  graph->serialize("./xmagic.xmodel");
#endif

  graph.release();
  if (z == 1)
    return test;
  return root;
  
}

xir::Graph *create_test_graph()
{

  // This information should come from a meta-data file, or command line, or other method
  std::vector<std::int32_t> inShape = {1, 224, 224, 3};
  std::vector<std::int32_t> outShape = {1, 224, 224, 3};
  std::string deviceName = "CPU";              // Call it what you want, this string is used by application to pick the subgraph to run
  std::string kernelName = "TestKernelRunner"; // Call it what you want, to pick the host-side controller code to run
  std::vector<char> mc_code = {'T', 'h', 'e', 'I', 'n', 's', 't', 'r', 'u', 'c', 't', 'i', 'o', 'n', 's'};
  std::vector<char> params = {'T', 'h', 'e', 'P', 'a', 'r', 'a', 'm', 'e', 't', 'e', 'r', 's'};

  // Create a graph object
  std::unique_ptr<xir::Graph> graph = xir::Graph::create("graph_test");

  // Add a "data-fix" operator to simulate an input
  auto data_input_attrs = xir::Attrs::create();
  data_input_attrs->set_attr<std::vector<std::int32_t>>("shape", inShape);
  data_input_attrs->set_attr<std::string>("data_type", "XINT8");
  auto data_input = graph->add_op("data-input",
                                  "data-fix",
                                  std::move(data_input_attrs),
                                  {});

  // Add custom operator so we can define our own output shape
  //   ignoring shape inference
  auto custom_attrs = xir::Attrs::create();
  custom_attrs->set_attr<std::vector<std::int32_t>>("shape", outShape);
  custom_attrs->set_attr<std::string>("data_type", "XINT8");
  auto custom_op = graph->add_op("data-output",
                                 "xmagic",
                                 std::move(custom_attrs),
                                 {{"input", {data_input}}});

  auto *root = graph->get_root_subgraph();

  // Every operator is now part of its own subgraph
  root->create_children();

  // Merge a set of operators into a new subgraph
  auto *input = root->merge_children({root->find_op("data-input")});
  auto *test = root->merge_children({root->find_op("data-output")});

  // Set the important graph meta-data that the runtime will need to read
  test->set_name("test_subgraph");
  test->set_attr<std::string>("device", deviceName);
  test->set_attr<std::string>("kernel", kernelName);
  test->set_attr<std::vector<char>>("mc_code", mc_code);
  test->set_attr<std::vector<char>>("params", params);

#if 0
  // Sanity check graph meta-data
  for (auto &subgraph : root->children_topological_sort())
  {
    std::cout << subgraph->get_name() << std::endl;

    if (subgraph->has_attr("device"))
      std::cout << "Device attribute: " << subgraph->get_attr<std::string>("device") << std::endl;
    if (subgraph->has_attr("kernel"))
      std::cout << "Kernel attribute: " << subgraph->get_attr<std::string>("kernel") << std::endl;
    if (subgraph->has_attr("mc_code"))
      std::cout << "This Subgraph has binary instructions" << std::endl;
    if (subgraph->has_attr("params"))
      std::cout << "This Subgraph has binary parameters / weights/ biases" << std::endl;

    auto inputTensors = subgraph->get_input_tensors();

    std::cout << "Inputs:" << std::endl;

    for (auto &inputTensor : inputTensors)
    {
      std::cout << inputTensor->get_name() << std::endl;
      auto shape = inputTensor->get_shape();
      for (auto extent : shape)
      {
        std::cout << "  " << extent << std::endl;
      }
    }

    auto outputTensors = subgraph->get_output_tensors();

    std::cout << "Outputs:" << std::endl;

    for (auto &outputTensor : outputTensors)
    {
      std::cout << outputTensor->get_name() << std::endl;
      auto shape = outputTensor->get_shape();
      for (auto extent : shape)
      {
        std::cout << "  " << extent << std::endl;
      }
    }
  }
  graph->serialize("./xmagic.xmodel");
#endif

  return graph.release();
}
