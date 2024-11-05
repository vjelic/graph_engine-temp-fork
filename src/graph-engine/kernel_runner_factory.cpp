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


#include <utility>
#include <functional>
#include <map>
#include <stdexcept>

#include "kernel_runners.hpp"
#include "kernel_runner_factory.hpp"

void KernelRunnerFactory::register_kernel_runners()
{
  static std::mutex mut;
  std::lock_guard lk{ mut };

  if (!initalized_)
  {
    #include "kernel_runners.cpp"
  }
  initalized_ = true;
}
 
void KernelRunnerFactory::register_kernel_runner(const std::string &kernelName, create_cb_t createFn)
{
  map_[kernelName] = createFn;
}

std::unique_ptr<KernelRunner> KernelRunnerFactory::create(const std::string &kernelName, Engine *engine, const xir::Subgraph *subgraph, xir::Attrs *attrs)
{
  const auto &createFn = map_[kernelName];
  if (!createFn)
  {
    throw std::runtime_error("Error: KernelFactory could not find a creator function for kernelName: " + kernelName);
  }
  return createFn(engine, subgraph, attrs);
}

// Must construct the static members
map_t KernelRunnerFactory::map_;
bool KernelRunnerFactory::initalized_ = false;
