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

#include "op_runner.hpp"

// This class will leverage a map from string to a callback function
using map_op_t = std::map<std::string, create_cb_op_t>;

/**
 * @class OpRunnerFactory
 *
 * @brief
 * OpRunnerFactory serves as a utility for constructing
 * OpRunner objects of various types.
 *
 * @details
 * OpRunnerFactor provides a register_op_runner static method.
 * That method should be called at the beginning of main, or when the graph-enigne library is loaded.
 * It should be called once for every specialization of OpRunner.
 *
 * The register_op_runner function is registering a callback function for that Kernel Runner.
 *
 * At runtime, OpRunnerFactory::create can be invoked to construct any number of registerd OpRunners.
 */
class OpRunnerFactory
{
public:
  /**
   * Register create callback functions for all OpRunners.
   */
  static void register_op_runners();
  
  /**
   * Register a creator callback function for a OpRunner.
   *
   * @param opName
   *  Name of the type of OpRunner that is to be created at runtime.
   * @param createFn
   *  A callback function for creating an instance of desired type of OpRunner.
   */
  static void register_op_runner(const std::string &opName, create_cb_op_t createFn);

  /**
   * Create an instance of the specified OpRunner.
   *
   * @param opName
   *  Name of the type of OpRunner that is to be created at runtime.
   * @param engine
   *  This is a pointer to a global thread pool used for submitting execution requests.
   * @param op
   *  This is a pointer to a op. A op is an object that aggregates
   *    the necessary meta data needed by a OpRunner.
   * @return
   *  Unique pointer to the newly created OpRunner of desired type.
   */
  static std::unique_ptr<OpRunner> create(const std::string &opName, Engine *engine, const xir::Op *op, xir::Attrs *attrs);

protected:
  static map_op_t map_;
  static bool initalized_;
};
