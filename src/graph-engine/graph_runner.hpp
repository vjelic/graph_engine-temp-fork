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
#include <algorithm>
#include <mutex>

#include <xir/graph/graph.hpp>
#include <vart/runner_ext.hpp>
#include <xrt/xrt_device.h>
#include <glog/logging.h>
#include <experimental/xrt_xclbin.h>
#include <experimental/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include "vitis/ai/env_config.hpp"
#include "graph-engine/create_graph_runner.hpp"
#include "graph-engine/qos.hpp"

#ifdef _WIN32
#include <vitis/ai/tracelogging.hpp>
#endif
DEF_ENV_PARAM(DEBUG_GRAPH_RUNNER, "0")
DEF_ENV_PARAM(DEBUG_INTERNAL_XCLBIN, "0")
DEF_ENV_PARAM(ENABLE_IPU_AIE_PROFILING, "0")
constexpr size_t MAX_CONTEXTS = 8;

std::string get_xclbin_file(const xir::Subgraph* subgraph, std::string&);

bool is_pdi_enabled(const xir::Subgraph* subgraph);

std::string get_xclbin_kernelName(xrt::xclbin& xclbin);

bool match_one_xclbin_xmodel(const xrt::xclbin* xrt_xclbin, const xir::Subgraph* subgraph);
 // One XrtContext will be shared by all Runners
class XrtContext {

public:
    static XrtContext& GetInstance() {
        static XrtContext instance(0); // Guaranteed to be destroyed.
                                            // Constructed on first use.
#ifdef _WIN32
        TL_TRACE("XrtContext& GetInstance") << &instance;
#endif
        return instance;
    }

    xrt::device& GetDevice() {
        std::unique_lock<std::mutex> lock(ctxIdx_mtx_);
        return device_;
    };
    xrt::hw_context& GetContext(int ctx_idx) { 
       std::unique_lock<std::mutex> lock(ctxIdx_mtx_); 
       return hw_contexts_[ctx_idx]; 
       };
    xrt::kernel& GetKernel(int ctx_idx, std::string ker_name) {
        std::unique_lock<std::mutex> lock(ctxIdx_mtx_);
        std::tuple<int, std::string> tup(ctx_idx, ker_name);
        return kernels_[tup];
       };
    xrt::xclbin& GetXclbin(int ctx_idx) { 
      std::unique_lock<std::mutex> lock(ctxIdx_mtx_);
      return xclbins_[ctx_idx]; 
      };

    static std::string find_right_xclbin(std::string& dir, const xir::Subgraph* subgraph) {
        std::string empty{};
        if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
            for (auto const& fnm : std::filesystem::recursive_directory_iterator(dir)) {
                if (std::filesystem::is_regular_file(fnm) && fnm.path().extension() == ".xclbin") {
                    xrt::xclbin xrt_xclbin = xrt::xclbin(fnm.path().string());
                    auto match = match_one_xclbin_xmodel(&xrt_xclbin, subgraph);
                    if (match) {
                        return fnm.path().string();
                    }
                }
            }
        }
        return empty;
    }
    void getorcreateNewContext(int ctx_idx, const xir::Subgraph* subgraph, std::map<std::string, std::uint32_t>& qos, std::string xclbin_file = get_env_var("XLNX_VART_FIRMWARE", ""), const std::vector<char>& xclbin_binary_data = std::vector<char>())
    { 
        std::unique_lock<std::mutex> lock(ctxIdx_mtx_); 
        bool isCtxExists = false;
        bool en_pdi = is_pdi_enabled(subgraph);
        if (std::find(ctx_indices_.begin(), ctx_indices_.end(), ctx_idx) != ctx_indices_.end())
        {
            auto & a_qos = aggregated_qos_[ctx_idx];
            if (qos.count("gops")) {
                a_qos["gops"] += qos["gops"];
            }
            if (qos.count("latency")) {
                a_qos["latency"] += qos["latency"];
            }
            if (feature_qos_update_) {
                hw_contexts_[ctx_idx].update_qos(a_qos);
            }

            isCtxExists = true;
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << "graph-runner share ctx_idx: " << ctx_idx;

            //# Create kernel objetcs for all PDI subgraphs
            std::tuple<int, std::string> tup(ctx_idx, "");
            if (en_pdi) {
                auto pdi_subgraphs = subgraph->children_topological_sort();
                std::for_each(pdi_subgraphs.cbegin(), pdi_subgraphs.cend(), [&](const xir::Subgraph* pdi_subgraph) {
                    if ((pdi_subgraph->get_attr<std::string>("type") == "PDI")) {
                        std::string pdi_ker_name = pdi_subgraph->get_attr<std::string>("name");
                        ker_names.push_back(pdi_ker_name);
                        tup = std::make_tuple(ctx_idx, pdi_ker_name);
                        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                            << "graph-runner share context and create kernel for: " << pdi_ker_name;
                        kernels_[tup] = xrt::kernel(hw_contexts_[ctx_idx], pdi_ker_name);
                    }
                });
            }

            return;
        }
        assert(isCtxExists == false);

        //auto xclbinFnm = get_xclbin_file(subgraph, xclbin_location_);
        std::string xclbinFnm;

        if (xclbin_binary_data.empty())
        {
            xclbin_location_ = xclbin_file;
            xclbinFnm = get_xclbin_file(subgraph, xclbin_location_);
        }
       

        ctx_indices_.push_back(ctx_idx);

        if (xclbin_binary_data.empty())
        {
            xclbins_[ctx_idx] = xrt::xclbin(xclbinFnm);
        }
        else
        {
            xclbins_[ctx_idx] = xrt::xclbin(xclbin_binary_data);
        }

        device_.register_xclbin(xclbins_[ctx_idx]);

        if (qos.empty()) {
            hw_contexts_[ctx_idx] = xrt::hw_context(device_, xclbins_[ctx_idx].get_uuid());
    
        }
        else {
            hw_contexts_[ctx_idx] = xrt::hw_context(device_, xclbins_[ctx_idx].get_uuid(), qos);
            aggregated_qos_[ctx_idx] = qos;
        }


        //# Create kernel objetcs for all PDI subgraphs
        std::tuple<int, std::string> tup(ctx_idx, "");
        if (en_pdi) {
            auto pdi_subgraphs = subgraph->children_topological_sort();
            std::for_each(pdi_subgraphs.cbegin(), pdi_subgraphs.cend(), [&](const xir::Subgraph* pdi_subgraph) {
                if ((pdi_subgraph->get_attr<std::string>("type") == "PDI")) {
                    std::string pdi_ker_name = pdi_subgraph->get_attr<std::string>("name");
                    ker_names.push_back(pdi_ker_name);
                    tup = std::make_tuple(ctx_idx, pdi_ker_name);
                    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                        << "graph-runner doesn't share context and create pdi kernel for: " << pdi_ker_name;
                    kernels_[tup] = xrt::kernel(hw_contexts_[ctx_idx], pdi_ker_name);
                }
            });
        }
        else {
            //# Kernel object
            auto kernel_name = get_xclbin_kernelName(xclbins_[ctx_idx]);
            ker_names.push_back(kernel_name);
            tup = std::make_tuple(ctx_idx, kernel_name);
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << "graph-runner doesn't share context and create kernel for: " << kernel_name;
            kernels_[tup] = xrt::kernel(hw_contexts_[ctx_idx], kernel_name);
        }

#ifdef _WIN32
        TL_TRACE("new HW context created ctx_idx = ") << ctx_idx;
#endif
    }
    int get_reserved_ctx_idx() {
        std::unique_lock<std::mutex> lock(ctxIdx_mtx_);
        return reserved_ctx_idx_--;
    }
    int get_reserved_ctx_idx_limit() {
        std::unique_lock<std::mutex> lock(ctxIdx_mtx_);
        if(ctx_indices_.size() < MAX_CONTEXTS)
          return reserved_ctx_idx_--;
        else
        {
          int min_ref_cnt = get_ref_cnt(ctx_indices_[0]);
          int min_cnt_ctx_idx = ctx_indices_[0];

          for(int i=0; i<ctx_indices_.size(); i++)
          {
            if(get_ref_cnt(ctx_indices_[i])<min_ref_cnt)
            {
              min_ref_cnt = get_ref_cnt(ctx_indices_[i]);
              min_cnt_ctx_idx = ctx_indices_[i];
            }
          }
          return min_cnt_ctx_idx;
        }
    }

    void inc_ref_cnt(int ctx_idx) {
        std::unique_lock<std::mutex> lock(ctxIdx_mtx_);
        ctx_ref_cnt_[ctx_idx]++;
    }
    void dec_ref_cnt(int ctx_idx, std::map<std::string, std::uint32_t>& qos) {
        std::unique_lock<std::mutex> lock(ctxIdx_mtx_);
        ctx_ref_cnt_[ctx_idx]--;
        if (ctx_ref_cnt_[ctx_idx]) {
            auto& a_qos = aggregated_qos_[ctx_idx];
            if (qos.count("gops")) {
                a_qos["gops"] -= qos["gops"];
            }
            if (qos.count("latency")) {
                a_qos["latency"] -= qos["latency"];
            }
            if (feature_qos_update_) {
                hw_contexts_[ctx_idx].update_qos(a_qos);
            }
        }
        else {
            aggregated_qos_.erase(ctx_idx);
            for (auto& i : ker_names) {
                std::tuple<int, std::string> tup(ctx_idx, i);
                kernels_.erase(tup);
            }
            //TODO remove check here, this is hack for vaitrace
            auto trace = (bool)stoi(get_env_var("EN_VAITRACE", "0"));
            if (!trace) {
                hw_contexts_.erase(ctx_idx);
                xclbins_.erase(ctx_idx);

                auto it = std::find(ctx_indices_.begin(), ctx_indices_.end(), ctx_idx);
                if (it == ctx_indices_.end())
                {
#ifdef _WIN32
                    TL_TRACE("Internal error: ctx_indices_ mismatch!!!") << ctx_idx;
#endif
                    throw std::runtime_error("Internal error: ctx_indices_ mismatch");
                }
                ctx_indices_.erase(it);
            }
        }
        return;
    }

private:
    void XrtFeatureCheck() {
#ifdef _WIN32
        auto h = ::LoadLibraryExA("xrt_coreutil.dll", NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (h != NULL) {
            auto p = ::GetProcAddress(h, "xrtVersionFeature");
            if (p != NULL) {
                    feature_qos_update_ = true; // All XRT versions that support the xrtVersionFeature API also support the update_qos API
            }
            ::FreeLibrary(h);
        }
#endif
    }
    XrtContext( int dev_idx = 0)
    {
        XrtFeatureCheck();
        device_ = xrt::device(dev_idx);
        
    }

    ~XrtContext() {
#ifdef _WIN32
        TL_TRACE("XrtContext::~XrtContext") << ctx_ref_cnt_.size();
#endif
    }

public:
    XrtContext(const XrtContext&) = delete;
    XrtContext& operator=(const XrtContext&) = delete;


protected:
    std::uint32_t get_ref_cnt(int ctx_idx) {
        return ctx_ref_cnt_[ctx_idx];
    }
    xrt::device device_;
    std::vector<int> ctx_indices_;
    std::unordered_map<int, xrt::xclbin> xclbins_;
    std::unordered_map<int, xrt::hw_context> hw_contexts_;
    std::map<std::tuple<int, std::string>, xrt::kernel> kernels_;
    std::vector<std::string> ker_names;
    std::unordered_map<int, std::uint32_t> ctx_ref_cnt_{};
    std::unordered_map<int, std::map<std::string, std::uint32_t>> aggregated_qos_{};
    std::mutex ctxIdx_mtx_;
    int reserved_ctx_idx_{ -1 };
    std::string xclbin_location_{};
    bool feature_qos_update_{ false }; //TODO, XRT Linux is not available for qos, could turn it on once it is enabled
};

/**
 * @class GraphRunner
 *
 * @brief
 * GraphRunner is a concrete implementation of vart::RunnerExt.
 * It is used to provide clients with VART inference APIs.
 *
 * @details
 * GraphRunner provides clients with VART APIs such as execute_async, and wait.
 * A client can create N threads each with its own GraphRunner for running jobs in parallel.
 */
class GraphRunner : public vart::RunnerExt
{
public:
  /**
   * Construct a GraphRunner object from an XMODEL file.
   *
   * @param xmodel
   *  File path of XMODEL from which to construct this graph runner.
   * @param attrs
   *  Pointer to xir::Attrs. This is used for sharing metadata across all runners in the graph.
   */
    GraphRunner(const std::string& xmodel, xir::Attrs * attrs = nullptr, bool bypass_pad = false);

  /**
   * Construct a GraphRunner object from an xir::Graph.
   *
   * @param graph
   *  Pointer to xir::Graph.
   * @param attrs
   *  Pointer to xir::Attrs. This is used for sharing metadata across all runners in the graph.
   */
  GraphRunner(const xir::Graph *graph, xir::Attrs* attrs = nullptr, bool bypass_pad = false);

  /**
   * Construct a GraphRunner object from a root subgraph.
   *
   * @param root
   *  Pointer to the root subgraph inside of an XMODEL.
   * @param attrs
   *  Pointer to xir::Attrs. This is used for sharing metadata across all runners in the graph.
   */
  GraphRunner(const xir::Subgraph *root, xir::Attrs* attrs = nullptr, bool bypass_pad = false);

  /**
   * Destroy GraphRunner object.
   */
  virtual ~GraphRunner();

  /**
   * Submit this GraphRunners' run fuction to the task pool.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   * @return
   *  Pair of job_id and status. The job_id is used to reference this execution later in wait().
   *  Status of 0 indicates that the job was submittied successfully.
   */
  virtual std::pair<uint32_t, int> execute_async(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs) override;

  /**
   * Calculate the priority of this graph runner based on QoS and performance preference.
   *
   * @param inputs
   *  Attributes
   * @param outputs
   *  QoS map
   */
  void parse_qos_args_and_efficient_mode(xir::Attrs* attrs);

  /**
   * Wait for specified job to complete.
   *
   * @param job_id
   *  The numerical job identifier to wait on.
   * @param timeout
   *  The timout in milliseconds to wait before aborting. A value of negative 1 will wait indefinitely.
   * @return
   *  Returns 0 on success.
   */
  virtual int wait(int job_id, int timeout) override;

  /**
   * Run inference for this GraphRunner.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   */
  void run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs);

  /**
   * Submit this GraphRunners' run fuction to the task pool, and wait for job completion.
   *
   * @param inputs
   *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
   * @param outputs
   *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
   * @param timeout
   *  Optional. The timout in milliseconds to wait before aborting. A value of negative 1 will wait indefinitely.
   * @return
   *  Pair of job_id and status. The job_id is used to reference this execution later in wait().
   *  Status of 0 indicates that the job was submittied successfully.
   */
  virtual std::pair<uint32_t, int> execute(
      const std::vector<vart::TensorBuffer *> &input,
      const std::vector<vart::TensorBuffer *> &output, int timeout = -1);

  /**
   * Get this GraphRunner's input_tensors_.
   *
   * @return
   *  Vector of xir::Tensor pointers from which the number of inputs and their shapes can be determined.
   */
  virtual std::vector<const xir::Tensor *> get_input_tensors() override;

  /**
   * Get this GraphRunner's output_tensors_.
   *
   * @return
   *  Vector of xir::Tensor pointers from which the number of outputs and their shapes can be determined.
   */
  virtual std::vector<const xir::Tensor *> get_output_tensors() override;

  /**
   * Get this GraphRunner's input tensor buffers.
   *
   * @return
   *  Vector of xir::TensorBuffer pointers where this GraphRunner will fetch its inputs.
   */
  virtual std::vector<vart::TensorBuffer *> get_inputs() override;

  /**
   * Get this GraphRunner's output tensor buffers.
   *
   * @return
   *  Vector of xir::TensorBuffer pointers where this GraphRunner will deposit its outputs.
   */
  virtual std::vector<vart::TensorBuffer *> get_outputs() override;

  /**
   * Construct and return TensorBuffers for this GraphRunner.
   *
   * @return
   *  Vector of unique_ptr of TensorBuffers.
   */
  virtual std::vector<std::unique_ptr<vart::TensorBuffer>> create_inputs();

  /**
   * Construct and return TensorBuffers for this GraphRunner.
   *
   * @return
   *  Vector of unique_ptr of TensorBuffers.
   */
  virtual std::vector<std::unique_ptr<vart::TensorBuffer>> create_outputs();

  /**
   * Get this GraphRunner's input scale factors.
   *
   * @return
   *  Vector of floats representing the scale factor for each input
   */
  std::vector<float> get_input_scale_factors();

  /**
   * Get this GraphRunner's output scale factors.
   *
   * @return
   *  Vector of floats representing the scale factor for each output
   */
  std::vector<float> get_output_scale_factors();

  /**
   * use xir::Attrs to pass info to RunnerExt at runtime.
   * return 0: good; 1: error, RunnerExt cannot find any useful info in Attrs
   */
  virtual int set_run_attrs(std::unique_ptr<xir::Attrs>&) override;

private:
  const xir::Subgraph *graph_;
  xir::Attrs *attrs_;
  static Engine *engine_;
  std::vector<std::unique_ptr<KernelRunner>> kernel_runners_;
  std::vector<std::vector<vart::TensorBuffer *>> tensor_buffers_raw_;
  std::vector<std::vector<std::unique_ptr<vart::TensorBuffer>>> tensor_buffers_;
  bool async_exec_ {true};
  int ctx_idx_{ -1 };
  bool context_inner_;
  bool attrs_inner_{ false };
  std::map<std::string, std::uint32_t> merged_qos_{};
  XrtContext& xrt_context_;
};
