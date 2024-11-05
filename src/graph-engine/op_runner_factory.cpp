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

#include "op_runners.hpp"
#include "op_runner_factory.hpp"

void OpRunnerFactory::register_op_runners()
{
  static std::mutex mut;
  std::lock_guard lg{ mut };

  if (!initalized_)
  {
    #include "op_runners.cpp"
  }
  initalized_ = true;
}
 
void OpRunnerFactory::register_op_runner(const std::string &opName, create_cb_op_t createFn)
{
  map_[opName] = createFn;
}

std::unique_ptr<OpRunner> OpRunnerFactory::create(const std::string &opName, Engine *engine, const xir::Op *op, xir::Attrs *attrs)
{
  const auto &createFn = map_[opName];
  if (!createFn)
  {
    throw std::runtime_error("Error: OpRunnerFactory could not find a creator function for opName: " + opName);
  }
  return createFn(engine, op, attrs);
}

// Must construct the static members
map_op_t OpRunnerFactory::map_;
bool OpRunnerFactory::initalized_ = false;
