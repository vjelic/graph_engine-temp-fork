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

#include "create_test_graph.hpp"
#include <vart/runner_ext.hpp>

TEST(VartRunnerTest, VartRunnerBasic)
{
  const xir::Subgraph *root = create_test_graph(0);

  auto attrs = xir::Attrs::create();
  auto graphRunner = vart::RunnerExt::create_runner(root, attrs.get());

  auto inputs = graphRunner->get_inputs();
  auto outputs = graphRunner->get_outputs();

  for (auto &input : inputs)
  {
    // Can fill inputs, but they are zero by default
  }

  auto job = graphRunner->execute_async(inputs, outputs);

  graphRunner->wait(job.first, -1);

  for (auto &output : outputs)
  {
    uint64_t ptrX;
    size_t size;
    std::tie(ptrX, size) = output->data();

    auto ptr = (int8_t *)ptrX;

    for (int i = 0; i < size; i++)
      EXPECT_EQ(1, ptr[i]) << "Buffers differ at index " << i;
  }
}
