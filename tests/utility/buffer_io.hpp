
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

#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <fstream>
#include <cstring>
#include <vart/tensor_buffer.hpp>
#include <xir/tensor/tensor.hpp>
#include "graph-engine/utility.hpp"

// Copy values from binary files into input buffer, expecting values are raw binary
void init_buf(vart::TensorBuffer *input, std::string &filename)
{
  uint64_t buff;
  size_t bytesize;
  std::tie(buff, bytesize) = input->data();

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);

  ASSERT_TRUE(ifs.is_open()) << "Failed To Open File";
  if (input->get_tensor()->has_attr("ddr_addr"))
  {
     ifs.seekg(input->get_tensor()->get_attr<int32_t>("ddr_addr"));
  }

  if (input->get_tensor()->has_attr("stride"))
  {
      bytesize = input->get_tensor()->get_attr<std::vector<std::int32_t>>("stride")[0];
  }
  ifs.read((char *)buff, bytesize);
}

// Copy values from binary files into input buffer, expecting values are raw binary
void init_buf(void *buff, size_t bytesize, std::string &filename)
{
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);

  ASSERT_TRUE(ifs.is_open()) << "Failed To Open File";

  ifs.read((char *)buff, bytesize);
}

// Compare values from binary file to output buffer, expecting values are raw binary
void comp_buf(vart::TensorBuffer *output, std::string &filename)
{
  uint64_t buff;
  size_t bytesize;
  std::tie(buff, bytesize) = output->data();

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);

  ASSERT_TRUE(ifs.is_open()) << "Failed To Open File";

  std::istreambuf_iterator<char> it(ifs), end;

  auto numElements = output->get_tensor()->get_element_num();
  for (int idx = 0; idx < numElements; idx++)
  {
    ASSERT_EQ(*it, ((char *)buff)[idx]) << "Buffers differ at index " << idx;
    it++;
  }
}

// Maybe fix this someday... this is horrible... it is 2am right now...
// Compare values from binary file to output buffer, expecting values are raw float
void comp_buf(void *buff, size_t bytesize, std::string &filename)
{

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);

  ASSERT_TRUE(ifs.is_open()) << "Failed To Open File";

  char *gold = new char[bytesize]; 

  ifs.read(gold, bytesize);

  float* a = (float*)gold;
  float *b = (float*)buff;

  for(int i = 0; i < bytesize/sizeof(float); i++) {
    ASSERT_EQ(a[i], b[i]) << "Buffers differ at index " << i;
  }

}

// Dump a buffer's binary contents to a binary file
void dump_buf(vart::TensorBuffer *buffer, std::string &filename)
{
  uint64_t buff;
  size_t bytesize;
  std::tie(buff, bytesize) = buffer->data();

  std::ofstream ofs(filename, std::ios::binary);

  ASSERT_TRUE(ofs.is_open()) << "Failed To Open File";

  ofs.write(reinterpret_cast<char *>(buff), bytesize);
}
