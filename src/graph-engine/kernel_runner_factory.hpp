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


#pragma once
#include <memory>
#include <string>
#include <utility>
#include <functional>
#include <map>
#include <stdexcept>

#include "kernel_runner.hpp"

// This class will leverage a map from string to a callback function
using map_t = std::map<std::string, create_cb_t>;

/**
 * @class KernelRunnerFactory
 *
 * @brief
 * KernelRunnerFactory serves as a utility for constructing
 * KernelRunner objects of various types.
 *
 * @details
 * KernelRunnerFactor provides a register_kernel_runner static method.
 * That method should be called at the beginning of main, or when the graph-enigne library is loaded.
 * It should be called once for every specialization of KernelRunner.
 *
 * The register_kernel_runner function is registering a callback function for that Kernel Runner.
 *
 * At runtime, KernelFactory::create can be invoked to construct any number of registerd KernelRunners.
 */
class KernelRunnerFactory
{
public:
  /**
   * Register create callback functions for all KernelRunners.
   */
  static void register_kernel_runners();
  
  /**
   * Register a creator callback function for a KernelRunner.
   *
   * @param kernelName
   *  Name of the type of KernelRunner that is to be created at runtime.
   * @param createFn
   *  A callback function for creating an instance of desired type of KernelRunner.
   */
  static void register_kernel_runner(const std::string &kernelName, create_cb_t createFn);

  /**
   * Create an instance of the specified KernelRunner.
   *
   * @param kernelName
   *  Name of the type of KernelRunner that is to be created at runtime.
   * @param engine
   *  This is a pointer to a global thread pool used for submitting execution requests.
   * @param subgraph
   *  This is a pointer to a subgraph. A subgraph is an object that aggregates
   *    the necessary meta data needed by a KernelRunner.
   * @return
   *  Unique pointer to the newly created KernelRunner of desired type.
   */
  static std::unique_ptr<KernelRunner> create(const std::string &kernelName, Engine *engine, const xir::Subgraph *subgraph, xir::Attrs *attrs);

protected:
  static map_t map_;
  static bool initalized_;
};
