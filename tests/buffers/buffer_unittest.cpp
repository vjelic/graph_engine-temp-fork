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
#include <gtest/gtest.h>

#include "graph-engine/graph-engine.hpp"

TEST(BufferTest, HostBuffer)
{

  std::vector<std::int32_t> shape = {1, 224, 224, 3};               // NHWC
  xir::DataType dataType(xir::DataType::XINT, sizeof(int8_t) * 8u); // INT8
  auto tensorPtr = xir::Tensor::create("input", shape, dataType);

  HostBuffer<int8_t> buffer(tensorPtr.get(), tensorPtr->get_data_size(), 0);

  uint64_t data;
  size_t size;

  std::tie(data, size) = buffer.data();

  std::cout << "data: " << (void *)data << " size: " << size << std::endl;

  EXPECT_TRUE((void*)data != nullptr);
}
