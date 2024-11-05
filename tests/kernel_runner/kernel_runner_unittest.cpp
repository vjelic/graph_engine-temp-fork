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

#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include "graph-engine/graph-engine.hpp"
#include "graph-engine/kernel_runner_factory.hpp"

#include "create_test_graph.hpp"

#include <algorithm>

class ParameterizedTestFixture : public ::testing::TestWithParam<int> {};

TEST_P(ParameterizedTestFixture, KernelRunnerTest)
{

  int numRunners = GetParam();

  // Register all kernel runners
  KernelRunnerFactory::register_kernel_runners();

  Engine engine;

  const xir::Subgraph *sg = create_test_graph(1);

  std::vector<std::unique_ptr<KernelRunner>> kernel_runners_;
  std::vector<std::vector<const xir::Tensor *>> input_tensors_;
  std::vector<std::vector<const xir::Tensor *>> output_tensors_;
  std::vector<std::vector<vart::TensorBuffer *>> tensor_buffers_raw_;
  std::vector<std::vector<std::unique_ptr<vart::TensorBuffer>>> tensor_buffers_;

  for (int i = 0; i < numRunners; i++)
    kernel_runners_.emplace_back(KernelRunnerFactory::create(sg->get_attr<std::string>("kernel"), &engine, sg, nullptr));

  for (auto &kernel_runner : kernel_runners_)
  {
    input_tensors_.emplace_back(kernel_runner->get_input_tensors());
    output_tensors_.emplace_back(kernel_runner->get_output_tensors());
    tensor_buffers_.emplace_back();

    for (auto &input_tensor : input_tensors_.back())
    {
      tensor_buffers_.back().emplace_back(
          std::make_unique<HostBuffer<int8_t>>(input_tensor, input_tensor->get_data_size(), 0));
    }
  }

  tensor_buffers_.emplace_back();
  for (auto &output_tensor : output_tensors_.back())
  {
    tensor_buffers_.back().emplace_back(std::make_unique<HostBuffer<int8_t>>(output_tensor, output_tensor->get_data_size(), 0));
  }

  for (auto &v : tensor_buffers_)
  {
    tensor_buffers_raw_.emplace_back();
    std::transform(v.begin(), v.end(), std::back_inserter(tensor_buffers_raw_.back()), [](std::unique_ptr<vart::TensorBuffer> &sp)
                   { return sp.get(); });
  }

  ASSERT_EQ(tensor_buffers_.size(), kernel_runners_.size() + 1) << "Vectors are of unequal length";

  for (int i = 0; i < kernel_runners_.size(); i++)
  {
    kernel_runners_[i]->execute(tensor_buffers_raw_[i], tensor_buffers_raw_[i + 1]);
  }

  auto &output_tensor_buffers_ = tensor_buffers_raw_.back();

  for (auto &tensor_buffer : output_tensor_buffers_)
  {
    auto &result = HostBuffer<int8_t>::get_vector(tensor_buffer);
    auto golden = std::vector<int8_t>(tensor_buffer->get_tensor()->get_data_size(), numRunners);
    EXPECT_TRUE(result == golden);
  }
}

INSTANTIATE_TEST_SUITE_P(
  kernel_runner_tests,
  ParameterizedTestFixture,
  ::testing::Range(1, 11)
);
