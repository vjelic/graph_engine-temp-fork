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

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <math.h>
#include <cstring>
#include <filesystem>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include <xir/graph/graph.hpp>
#include <vart/runner_ext.hpp>

#include "graph-engine/kernel_runner_factory.hpp"
#include "graph-engine/op_runner_factory.hpp"
#include "graph-engine/create_graph_runner.hpp"
#include "graph-engine/create_dpu_runner.hpp"
#include "graph-engine/utility.hpp"
#include "graph-engine/mep_xclbins_table.hpp"

#include "graph-engine/graph_runner.hpp"

#include "graph-engine/utils/FixFloat.hpp"

#include "vitis/ai/profiling.hpp"

#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <experimental/xrt_hw_context.h>

#ifdef VART_IN_BUILD_TREE
#include <vitis/ai/trace.hpp>
#include <vitis/ai/traceclass.hpp>
#else
#include <vart/trace/trace.hpp>
#include <vart/trace/traceclass.hpp>
#endif

// This library does not even work in Linux
// So don't include it if not windows
#ifdef _WIN32
#include <cfgmgr32.h> // CM_Get_Device_ID_List_Size
#include <regex>
#include <vitis/ai/tracelogging.hpp>
#endif

DEF_ENV_PARAM(GET_WORKLOADONARCH_BY_EGOPS, "0");

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

// Static functions used only in this file
namespace
{

  inline float get_input_scale(const xir::Tensor *tensor) {
    ASSERT(tensor->has_attr("fix_point"), "This tensor does not have a scale factor");
    int fixpos = tensor->template get_attr<int>("fix_point");
    return std::exp2f(1.0f * (float)fixpos);
  }

    inline std::pair<float, int> get_input_scale_zero_point(const xir::Tensor *tensor) { 
      ASSERT(tensor->has_attr("scale"), "This tensor does not have a scale attr");
      ASSERT(tensor->has_attr("zero_point"), "This tensor does not have zero_point attr");
      
      auto zero_point = tensor->template get_attr<std::vector<int>>("zero_point");
      ASSERT(zero_point.size() == 1, "Zero point must be a vector of size 1");
      auto scale = tensor->template get_attr<std::vector<float>>("scale");
      ASSERT(scale.size() == 1, "Scale must be a vector of size 1");
      return {scale[0], zero_point[0]};
    }

  inline float get_output_scale(const xir::Tensor* tensor) {
    ASSERT(tensor->has_attr("fix_point"), "This tensor does not have a scale factor");
    int fixpos = tensor->template get_attr<int>("fix_point");
    return std::exp2f(-1.0f * (float)fixpos);
  }

  inline std::pair<float, int> get_output_scale_zero_point(const xir::Tensor *tensor) { 
    ASSERT(tensor->has_attr("scale"), "This tensor does not have a scale attr");
    ASSERT(tensor->has_attr("zero_point"), "This tensor does not have zero_point attr");
    auto scale = tensor->template get_attr<std::vector<float>>("scale");
    ASSERT(scale.size() == 1, "Scale must be a vector of size 1");
    auto zero_point = tensor->template get_attr<std::vector<int>>("zero_point");
    ASSERT(zero_point.size() == 1, "Zero point must be a vector of size 1");
    return { scale[0], zero_point[0] };
  }

  Engine *get_engine_instance()
  {
    static Engine *engine = new Engine;
    return engine;
  }
}
bool is_pdi_enabled(const xir::Subgraph* subgraph) {
    bool en_pdi = false;
    if (subgraph->has_attr("enable_pdi"))
        en_pdi = subgraph->get_attr<bool>("enable_pdi");
    auto debug_mode = (bool)stoi(get_env_var("XLNX_ENABLE_DEBUG_MODE", "0"));

    if ((subgraph->has_attr("enable_fast_pm")) && (subgraph->get_attr<bool>("enable_fast_pm")) && (!debug_mode)) {
        en_pdi = false;
    }
    return en_pdi;
}
std::string get_xclbin_kernelName(xrt::xclbin& xclbin) {
    std::string kernel_name;
    bool find = false;
    auto xclbin_kernels = xclbin.get_kernels();
    for (auto kernel : xclbin_kernels) {
        if (kernel.get_name().find("DPU") != std::string::npos) {
            kernel_name = kernel.get_name();
            find = true;
            break;
        }
    }
    if (!find)
        throw std::runtime_error("Couldn't find a correct kernel that contains DPU ");
    return kernel_name;
}
bool match_one_xclbin_xmodel(const xrt::xclbin* xrt_xclbin, const xir::Subgraph* subgraph) {
    uint64_t xmodel_fingerprint = 0;
    if (subgraph->has_attr("dpu_fingerprint")) {
        xmodel_fingerprint = subgraph->get_attr<uint64_t>("dpu_fingerprint");
    }

    uint64_t expect_xclbin = xmodel_fingerprint;
    // workaround for .122 docker
    if (xmodel_fingerprint == 576460752305570371) // .122 xmodel's fingerprint
        expect_xclbin = 576462972809687554;   // .122 xclbin's fingerprint

    uint64_t xclbin_fingerprint = -1;
    for (auto& aiep : xrt_xclbin->get_aie_partitions()) {
        xclbin_fingerprint = aiep.get_inference_fingerprint();
        if (xclbin_fingerprint == expect_xclbin) {
            std::string printFingerPrint = get_env_var("XLNX_VART_PRINT_FP");
            if (printFingerPrint == "TRUE") {
                std::cout << "Fingerprint of xclbin and subgraph: " << xclbin_fingerprint << std::endl;
            }
            return true;
        }
    }
    return false;
}

static std::string find_right_xclbin(const std::string& dir, const xir::Subgraph* subgraph) {
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

#ifdef _WIN32
// Phoenix device ID: 0x1502
// Strix device ID: 0x17f0
const std::regex device_IDs_regex(R"(PCI\\VEN_1022&DEV_(1502|17F0))");

/* This function takes the PCI Vendor ID and Device ID as input (device_id) and searches for the driver store associated with the specified device.
 * It uses the Windows Configuration Manager API (CM_Get_Device_ID_ListA, CM_Locate_DevNodeA, CM_Open_DevNode_Key, RegGetValueA) to interact with the system's hardware registry.
 * The found driver store path is returned in the value parameter.
 */
int find_driver_store_by_device_id(char* device_id, char* sub_key, char* val_name, DWORD data_type_flag, std::string& value) {
    ULONG len = 0;
    CONFIGRET ret = ::CM_Get_Device_ID_List_SizeA(&len, device_id, CM_GETIDLIST_FILTER_ENUMERATOR | CM_GETIDLIST_FILTER_PRESENT);  // len is in character
    if (ret != CR_SUCCESS) {
        return -1;
    }

    char* buffer = new char[len];
    ret = ::CM_Get_Device_ID_ListA(device_id, buffer, len, CM_GETIDLIST_FILTER_ENUMERATOR | CM_GETIDLIST_FILTER_PRESENT);
    if (ret != CR_SUCCESS) {
        delete[] buffer;
        return -2;
    }

    std::vector<std::string> vec_dev{};
    ULONG vec_len = 0;
    std::smatch sm;
    ULONG dev_len_start = 0;
    ULONG dev_len = 0;
    char dev_buffer[MAX_DEVICE_ID_LEN] = { 0 };
    do {
        std::string ms(&buffer[vec_len]);
        if (std::regex_search(ms, sm, device_IDs_regex)) {
            vec_dev.push_back(std::string(&buffer[vec_len]));
            dev_len_start = vec_len;
            dev_len = vec_dev.back().size() + 1;
            std::copy(buffer + dev_len_start, buffer + dev_len_start + dev_len, dev_buffer);
        }
        vec_len += ms.size() + 1;
    } while (vec_len < len);
    delete[] buffer;

    if (vec_dev.size() > 1) {
        return -100 - vec_dev.size();
    }

    DEVINST dnDevInst = 0;
    ret = ::CM_Locate_DevNodeA(&dnDevInst, dev_buffer, CM_LOCATE_DEVNODE_NORMAL);
    if (ret != CR_SUCCESS) {
        return -3;
    }

    HKEY key = 0;
    ret = ::CM_Open_DevNode_Key(dnDevInst, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &key, CM_REGISTRY_HARDWARE);
    if (ret != CR_SUCCESS) {
        return -4;
    }

    if (key) {
        DWORD get_data_len = 0;
        ret = RegGetValueA(key, sub_key, val_name, data_type_flag, NULL, NULL, &get_data_len);
        if (ERROR_SUCCESS != ret) {
            ret = ::RegCloseKey(key);
            return -5;
        }
        char* get_data = new char[get_data_len];
        ret = RegGetValueA(key, sub_key, val_name, data_type_flag, NULL, get_data, &get_data_len);
        if (ERROR_SUCCESS != ret) {
            delete[] get_data;
            ret = ::RegCloseKey(key);
            return -6;
        }
        value = std::string(get_data);
        delete[] get_data;
        ret = ::RegCloseKey(key);
    }

    return 0;
}

/* This function uses the previously mentioned find_driver_store_by_device_id to search for the driver store associated with a specific device identified as "IPU" (Inference Processing Unit) and stored under the registry key "IPU DriverStore Path".
 * It returns the path to the driver store.
 */
std::string find_ipu_driver_store() {
    std::string driver_store_path{};
    int ret = find_driver_store_by_device_id(TEXT("PCI"), TEXT("IPU DriverStore Path"), TEXT("IPUDriverStore"), RRF_RT_REG_SZ, driver_store_path);
    if (ret)
    {
        std::cerr << "find_driver_store_by_device_id() ret = " << ret << std::endl;
    }
    return driver_store_path;
}
#endif // #ifdef _WIN32

std::unordered_map<uint64_t, std::string> load_config_xclbin_file(const std::string& filename) {
    std::unordered_map<uint64_t, std::string> fingerprint_xclbinfilename;
    char delimiter = ':';
    std::ifstream inputFile(filename);

    if (inputFile.is_open()) {
        std::string line;
        while (std::getline(inputFile, line)) {
            std::istringstream ip(line);
            uint64_t fingerprint;
            std::string xclbinfilename;

            if (ip >> fingerprint >> delimiter && std::getline(ip, xclbinfilename)) {
                xclbinfilename.erase(0, xclbinfilename.find_first_not_of(" \t\r\n"));
                xclbinfilename.erase(xclbinfilename.find_last_not_of(" \t\r\n") + 1);
                fingerprint_xclbinfilename[fingerprint] = xclbinfilename;
            }
        }
        inputFile.close();
    }
    else {
        std::cerr << "Error opening config xclbin file: " << filename << std::endl;
    }

    return fingerprint_xclbinfilename;
}



/* This function is used to get the location of an xclbin file.
 * It checks if the xclbin_location variable is empty and, if so, proceeds to find the path of the xclbin file.
 * The path is obtained by first checking for an environment variable named "XLNX_VART_FIRMWARE".
 * If the environment variable is not set or does not contain a valid path, the find_ipu_driver_store function is called to search for the IPU driver store.
 * If both attempts fail to find a valid path, it sets the xclbinFnm variable to "C:\\Windows\\System32\\AMD".
 * The final path of the xclbin file is stored in the xclbin_location parameter.
 */
std::string get_xclbin_file(const xir::Subgraph* subgraph, std::string& xclbin_location) {

#ifdef _WIN32
    TL_TRACE("In get xclbin file function: 0: xclbin_location: ") << xclbin_location;
#endif
    uint64_t xmodel_fingerprint = 0;
    std::string xclbinFnm;
    std::string driver_store_path;
    if (subgraph->has_attr("dpu_fingerprint")) {
        xmodel_fingerprint = subgraph->get_attr<uint64_t>("dpu_fingerprint");
    }
#ifdef _WIN32
    TL_TRACE("In get xclbin file function: 1: xmodel_fingerprint: ") << xmodel_fingerprint;
#endif
#ifdef _WIN32
    if (!ENV_PARAM(DEBUG_INTERNAL_XCLBIN))
    {
        auto mep_xclbin_it = mep_xclbins_table.find(xmodel_fingerprint);
#ifdef _WIN32
        TL_TRACE("In get xclbin file function: 2: Looking in mep xclbins table:");
#endif
        if (mep_xclbin_it != mep_xclbins_table.end())
        {
            driver_store_path = find_ipu_driver_store();
#ifdef _WIN32
            TL_TRACE("In get xclbin file function: 3: driver store path: ") << driver_store_path;
#endif
            xclbinFnm = driver_store_path + "\\" + mep_xclbins_table.at(xmodel_fingerprint);
#ifdef _WIN32
            TL_TRACE("In get xclbin file function: 4: xclbinFnm mep loop: ") << xclbinFnm;
#endif
            return xclbinFnm;
        }
    }
#endif
    if (xclbin_location.empty()) {
#ifdef _WIN32
        xclbinFnm = get_env_var("XLNX_VART_FIRMWARE");
#ifdef _WIN32
        TL_TRACE("In get xclbin file function: 5: xclbinFnm env var: ") << xclbinFnm;
#endif
        if (xclbinFnm.empty()) {
            xclbinFnm = driver_store_path;
            std::string config_xclbin_filename = xclbinFnm + "\\config_xclbin.txt";
#ifdef _WIN32
            TL_TRACE("In get xclbin file function: 6: config xclbin fname: ") << config_xclbin_filename;
#endif
            if (std::filesystem::exists(config_xclbin_filename))
            {
                std::unordered_map<uint64_t, std::string> fingerprint_xclbinfilename_map = load_config_xclbin_file(config_xclbin_filename);
#ifdef _WIN32
                TL_TRACE("In get xclbin file function: 7: load config xclbin file: ");
#endif
                auto it = fingerprint_xclbinfilename_map.find(xmodel_fingerprint);

                if (it != fingerprint_xclbinfilename_map.end())
                {
                    std::string config_xclbin_file = xclbinFnm + "\\" + it->second;
                    xrt::xclbin xrt_xclbin = xrt::xclbin(config_xclbin_file);
                    auto match = match_one_xclbin_xmodel(&xrt_xclbin, subgraph);
                    if (match)
                        return config_xclbin_file;
                }
                
            }
            

            if (xclbinFnm.empty()) {
                xclbinFnm = std::string("C:\\Windows\\System32\\AMD");
#ifdef _WIN32
                TL_TRACE("In get xclbin file function: 8: xclbinFnm default path: ") << xclbinFnm;
#endif
            }
        }
#else
    std::string xclbinFnm = get_env_var("XLNX_VART_FIRMWARE", "C:\\Windows\\System32\\AMD");
#endif

    if (std::filesystem::is_regular_file(xclbinFnm) && std::filesystem::path(xclbinFnm).extension() == ".xclbin") {   // in case user set a particular xclbin file to use/test, which user should make sure it works with subgraph
        xrt::xclbin xrt_xclbin = xrt::xclbin(xclbinFnm);
        auto match = match_one_xclbin_xmodel(&xrt_xclbin, subgraph);
        if (!match) {
#ifdef _WIN32
            TL_TRACE("Warning: fingerprint of xclbin filectx_idx set by user ") << xclbinFnm << " doesn't match subgraph " << subgraph->get_name();
#endif
            std::string skipFingerPrintCheck = get_env_var("XLNX_VART_SKIP_FP_CHK");
            if (skipFingerPrintCheck != "TRUE") {
                std::cout << "Mismatch: xclbin filename: " << xclbinFnm << " subgraph: " << subgraph->get_name() << std::endl;
                throw std::runtime_error("Error: Fingerprint of xclbin does not match subgraph's fingerprint");
            }
        }
            xclbin_location = xclbinFnm;
#ifdef _WIN32
            TL_TRACE("In get xclbin file function: 9: xclbinFnm: ") << xclbinFnm;
#endif
    }
    else {
        auto find = find_right_xclbin(xclbinFnm, subgraph);
        if (find.empty())
            throw std::runtime_error("Error: No compatible xclbin file found.");
            else {
#ifdef _WIN32
                TL_TRACE("Found xclbin file: ") << find;
#endif
                xclbin_location = xclbinFnm;
            xclbinFnm = find;
    }
#ifdef _WIN32
        TL_TRACE("In get xclbin file function: 10: ") << xclbinFnm;
#endif    
    }
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "xclbin location: " << xclbinFnm;
        return xclbinFnm;
    }
    else if (std::filesystem::is_regular_file(xclbin_location) && std::filesystem::path(xclbin_location).extension() == ".xclbin") {   // in case user set a particular xclbin file to use/test, which user should make sure it works with subgraph
        xrt::xclbin xrt_xclbin = xrt::xclbin(xclbin_location);
        auto match = match_one_xclbin_xmodel(&xrt_xclbin, subgraph);
        if (!match) {
#ifdef _WIN32
            TL_TRACE("Warning: fingerprint of xclbin filectx_idx set by user ") << xclbin_location << " doesn't match subgraph " << subgraph->get_name();
#endif
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "xclbin location: " << xclbin_location;
#ifdef _WIN32
        TL_TRACE("In get xclbin file function: 11: xclbinFnm: ") << xclbin_location;
#endif
        return xclbin_location;
    }
    else {
        auto find = find_right_xclbin(xclbin_location, subgraph);
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "xclbin location: " << find;
        if (find.empty())
            throw std::runtime_error("Error: No compatible xclbin file found.");
        else {
#ifdef _WIN32
            TL_TRACE("Found xclbin file: ") << find;
#endif
#ifdef _WIN32
            TL_TRACE("In get xclbin file function: 12: xclbinFnm: ") << find;
#endif
            return find;
        }
    }
}

static void default_initialize_attrs(const xir::Subgraph* subgraph, xir::Attrs* attrs, std::map<std::string, std::uint32_t>& qos, XrtContext& xrt_context)
{
    // XrtContext manages HW context
    int ctx_idx0;
    if (attrs->has_attr("ctx_idx"))
        ctx_idx0 = attrs->get_attr<int>("ctx_idx");
    else
    {
        ctx_idx0 = xrt_context.get_reserved_ctx_idx_limit();
        attrs->set_attr<int>("ctx_idx", ctx_idx0);
        xrt_context.inc_ref_cnt(ctx_idx0);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
        << "graph-runner default initialize attrs use ctx_idx = " << ctx_idx0;
#ifdef _WIN32
    TL_TRACE("default_initialize_attrs, ctx_idx = ") << ctx_idx0;
#endif

    if (attrs->has_attr("xclbin_file") && attrs->has_attr("xclbin_raw_data")) {
	    throw std::runtime_error("User provided attrs can have only one of the attrs - either xclbin_file or xclbin_raw_data, any other scenario is not supported ");
    } 
    else if(attrs->has_attr("xclbin_file"))
    {
        xrt_context.getorcreateNewContext(ctx_idx0, subgraph, qos, (std::string)attrs->get_attr<std::string>("xclbin_file"));
        
    }
    else if (attrs->has_attr("xclbin_raw_data"))
    {
        
        std::vector<char> xclbin_data = attrs->get_attr<std::vector<char>>("xclbin_raw_data");
        xrt_context.getorcreateNewContext(ctx_idx0, subgraph, qos, "", xclbin_data);
    }
    else
    {
       xrt_context.getorcreateNewContext(ctx_idx0, subgraph, qos);        
    }

    attrs->set_attr<xrt::device*>("xrt_device", &(xrt_context.GetDevice()));
    attrs->set_attr<xrt::xclbin*>("xrt_xclbin", &(xrt_context.GetXclbin(ctx_idx0)));
    attrs->set_attr<xrt::hw_context*>("xrt_hw_context", &(xrt_context.GetContext(ctx_idx0)));
    //# Check for PDI enable
    bool en_pdi = is_pdi_enabled(subgraph);

    //# Create kernel objetcs for all PDI subgraphs
    if (en_pdi) {
        auto* xrt_xclbin = attrs->get_attr<xrt::xclbin*>("xrt_xclbin");
        auto* xrt_hw_context = attrs->get_attr<xrt::hw_context*>("xrt_hw_context");
        auto pdi_subgraphs = subgraph->children_topological_sort();
        std::string ker_name, attr_name;
        std::for_each(pdi_subgraphs.cbegin(), pdi_subgraphs.cend(), [&](const xir::Subgraph* pdi_subgraph) {
            if ((pdi_subgraph->get_attr<std::string>("type") == "PDI")) {
                ker_name = pdi_subgraph->get_attr<std::string>("name");
                attr_name = pdi_subgraph->get_name() + '_' + ker_name;
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << "graph-runner store kernel with name: " << attr_name;
                attrs->set_attr<xrt::kernel*>(attr_name, &(xrt_context.GetKernel(ctx_idx0, ker_name)));
            }
        });
    }
    else {
        auto kernel_name = get_xclbin_kernelName(xrt_context.GetXclbin(ctx_idx0));
        attrs->set_attr<xrt::kernel*>("xrt_kernel", &(xrt_context.GetKernel(ctx_idx0, kernel_name)));
    }

    vitis::ai::trace::add_info("dpu-controller", "cu_device_id", 0,
                              "cu_core_id", 0, "cu_batch", 1,
                              TRACE_VAR("DPU"), TRACE_VAR("IPU_DPU"),
                              "cu_fingerprint", 1);
    return;
}

static std::unordered_map<std::string, std::uint64_t> get_qos_from_subgraph(const xir::Subgraph* Subgraph)
{
    std::unordered_map<std::string, std::uint64_t> qos_in_subg;
    std::uint64_t workload = 0;
    std::uint64_t workload_on_arch = 0;

    std::uint64_t xmodel_fingerprint = 0;
    if (Subgraph->has_attr("dpu_fingerprint")) {
        xmodel_fingerprint = Subgraph->get_attr<std::uint64_t>("dpu_fingerprint");
    }

    if (Subgraph->has_attr("workload_on_arch") && (xmodel_fingerprint != 576460752305570371)) { // workload_on_arch is not valid for .122 xmodel
        workload_on_arch = Subgraph->get_attr<std::uint64_t>("workload_on_arch");
    }
    if (Subgraph->has_attr("workload")) {
        workload = Subgraph->get_attr<std::uint64_t>("workload");
    }
    if ( (workload_on_arch & workload) && (workload_on_arch < workload) ){ // both exists but workload_on_arch is smaller, then not valid
            workload_on_arch = 0;
    }
    qos_in_subg["workload"] = workload;
    qos_in_subg["workload_on_arch"] = workload_on_arch;

    if (Subgraph->has_attr("const_data_load_size")) {
        qos_in_subg["const_data_load_size"] = Subgraph->get_attr<std::uint64_t>("const_data_load_size");
    }
    if (Subgraph->has_attr("output_feature_save_size")) {
        qos_in_subg["output_feature_save_size"] = Subgraph->get_attr<std::uint64_t>("output_feature_save_size");
    }
    if (Subgraph->has_attr("input_feature_load_size")) {
        qos_in_subg["input_feature_load_size"] = Subgraph->get_attr<std::uint64_t>("input_feature_load_size");
    }

    return qos_in_subg;
}

GraphRunner::GraphRunner(const std::string &xmodel, xir::Attrs *attrs, bool bypass_pad)
  : GraphRunner(xir::Graph::deserialize(xmodel).release(), attrs, bypass_pad) {}

GraphRunner::GraphRunner(const xir::Graph *graph, xir::Attrs *attrs, bool bypass_pad)
  : GraphRunner(graph->get_root_subgraph(), attrs, bypass_pad) {}

GraphRunner::GraphRunner(const xir::Subgraph* root, xir::Attrs* attrs, bool bypass_pad)
    : attrs_(attrs), xrt_context_(XrtContext::GetInstance())
{
#ifdef _WIN32
    TL_TRACE_BLOCK("GraphRunner::GraphRunner instance") << this;
#endif

    bool equivalentXmodelFound = false;
    context_inner_ = true;
    ctx_idx_ = 0;
#ifdef _WIN32
    if (ENV_PARAM(DEBUG_GRAPH_RUNNER))
    {
        auto subgraphs_loop = root->children_topological_sort();
        for (auto subgraph : subgraphs_loop) {
            if (subgraph->has_attr("mc_code"))
            {
                std::vector<char> mc_code = subgraph->get_attr<std::vector<char> >("mc_code");
                std::hash<std::string> hash_mc_code;
                std::string mc_code_string(mc_code.begin(), mc_code.end());
                size_t hash_mc_code_val = hash_mc_code(mc_code_string);
                std::string xmodelToFind = std::to_string(hash_mc_code_val) + ".xmodel";
                std::string dir = "C:\\Windows\\System32\\AMD\\ipu_2";

                if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
                    for (auto const& fnm : std::filesystem::recursive_directory_iterator(dir)) {
                        if (std::filesystem::is_regular_file(fnm) && fnm.path().extension() == ".xmodel") {
                            std::string fnmxmodel = fnm.path().string();
                            if (fnmxmodel.find(xmodelToFind) != std::string::npos) {
                                const xir::Graph* graph_equivalent_xmodel = xir::Graph::deserialize(fnmxmodel).release();
                                const xir::Subgraph* root_equivalent_xmodel = graph_equivalent_xmodel->get_root_subgraph();
                                graph_ = root_equivalent_xmodel;
                                equivalentXmodelFound = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (equivalentXmodelFound)
                break;
        }
    }
#endif
    if (!equivalentXmodelFound)
        graph_ = root;

#ifdef _WIN32
    TL_TRACE("attrs = ") << attrs_;
#endif
    if (attrs)
    {
        if (attrs->has_attr("ctx_idx")) // user set ctx_idx, use xrt_context_ to create HW context
        {
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << "graph-runner create with ctx_idx: "<< attrs->get_attr<int>("ctx_idx");
            if (attrs->has_attr("xrt_device") && attrs->has_attr("xrt_xclbin") && attrs->has_attr("xrt_hw_context")) { // share attrs will include such params
                assert(attrs_->get_attr<bool>("default_initialize"));
                ctx_idx_ = attrs->get_attr<int>("ctx_idx");
            }
            else {
                assert(!attrs->has_attr("xrt_device") && !attrs->has_attr("xrt_xclbin") && !attrs->has_attr("xrt_hw_context"));
                ctx_idx_ = attrs->get_attr<int>("ctx_idx");
                if (ctx_idx_ < 0) {
#ifdef _WIN32
                    TL_TRACE("All negative context_ids are reserved, cannot be used by user!");
#endif
                    throw std::runtime_error("All negative context_ids are reserved, cannot be used by user!");
                }
                attrs_->set_attr<bool>("default_initialize", true);
            }
            xrt_context_.inc_ref_cnt(ctx_idx_);
        }
        else // no ctx_idx set in attrs
        {
            assert(!attrs->has_attr("ctx_idx"));
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << "graph-runner create without ctx_idx";
            if (attrs->has_attr("xrt_device") && attrs->has_attr("xrt_xclbin") && attrs->has_attr("xrt_hw_context")) // get all 3, no need to create HW context
            {
                context_inner_ = false;
                attrs->set_attr<bool>("default_initialize", false); // don't know context-id, default -1, cannot track the context usage.
            }
            else if (attrs->has_attr("xrt_device") || attrs->has_attr("xrt_xclbin") || attrs->has_attr("xrt_hw_context")) // not get all 3, wrong
            {
#ifdef _WIN32
                TL_TRACE("xrt_device, xrt_xclbin or xrt_hw_context is missing from attributes!");
#endif
                throw std::runtime_error("xrt_device, xrt_xclbin or xrt_hw_context is missing from attributes");
            }
            else { // needs to create HW context, attrs->has_attr("ctx_idx") is false
                attrs->set_attr<bool>("default_initialize", true);
                ctx_idx_ = xrt_context_.get_reserved_ctx_idx_limit();
                attrs->set_attr<int>("ctx_idx", ctx_idx_);
                xrt_context_.inc_ref_cnt(ctx_idx_);
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << "graph-runner create without ctx_idx, generate ctx_idx: " << ctx_idx_;
#ifdef _WIN32
                TL_TRACE("Warning: no user setting context-id, use reserved");
#endif
            }
        }
        attrs_ = attrs;
    }
    else // no attrs from user
    {
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "graph-runner create without attrs";
        auto attrs = xir::Attrs::create();
        attrs->set_attr<bool>("default_initialize", true);
        attrs->set_attr<bool>("internal_attr_obj", true);
        attrs_inner_ = true;
        ctx_idx_ = xrt_context_.get_reserved_ctx_idx_limit();
        attrs->set_attr<int>("ctx_idx", ctx_idx_);
        xrt_context_.inc_ref_cnt(ctx_idx_);
        attrs_ = attrs.release();
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "graph-runner create without attrs, generate ctx_idx: " << ctx_idx_;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
        << "graph-runner create with pypass_pad value: " << bypass_pad;
    if (!attrs_->has_attr("bypass_pad"))
        attrs_->set_attr<bool>("bypass_pad", bypass_pad);
    else
        attrs_->set_attr<bool>("bypass_pad", bypass_pad || attrs_->get_attr<bool>("bypass_pad"));

  if (attrs_->has_attr("async_exec")) {
    async_exec_ = attrs_->get_attr<bool>("async_exec");
  }
  KernelRunnerFactory::register_kernel_runners();

  OpRunnerFactory::register_op_runners();

  std::vector<const xir::Subgraph*>subgraphs;
  if ((graph_->has_attr("device")) && (graph_->get_attr<std::string>("device") == "DPU")) {
      subgraphs.push_back(graph_);
  }
  else {
      subgraphs = graph_->children_topological_sort();
  }

  std::for_each(subgraphs.cbegin(), subgraphs.cend(), [&](const xir::Subgraph *subgraph)
  {
    // All subgraphs must have a device attribute
    if(subgraph->has_attr("device"))
    {
      if(!bypass_pad)
      {
        // If we are not bypassing CPU Subgraphs, construct them
        if(!attrs_->has_attr("bypass_cpu") || !attrs_->get_attr<bool>("bypass_cpu"))
        {
            if (subgraph->get_attr<std::string>("device") == "CPU")
            {
                auto ops = subgraph->get_ops();
                auto op = *next(ops.begin(), 0);
                if((subgraph->get_ops()).size()==1 && op->get_type() == "pad-fix")
                   this->kernel_runners_.emplace_back(KernelRunnerFactory::create("CpuKernelRunner", engine_, subgraph, attrs_));
            }
        }
      }

      // If we are not running Fake DPU, use standard DpuKernelRunner
      if(!attrs_->has_attr("fake_dpu") || !attrs_->get_attr<bool>("fake_dpu"))
      {
          if (subgraph->get_attr<std::string>("device") == "DPU") {
              // get QoS info from subgraph
              if(!ENV_PARAM(GET_WORKLOADONARCH_BY_EGOPS)) {
                if (subgraph->has_attr("workload")) { // change to workload_on_arch when subgraph has this attr
                    auto workload = subgraph->get_attr<std::uint64_t>("workload");
                    merged_qos_.emplace("gops", std::ceil(workload / 1000000000.0));
#ifdef _WIN32
                    TL_TRACE("gops from xmodel") << workload / 1000000000.0 << ", graph-runner: " << this << ", ctx_idx_ = " << ctx_idx_;
#endif                    
                }    
              } else {          
                std::unordered_map<std::string, std::uint64_t> all_qos = get_qos_from_subgraph(subgraph);
                merged_qos_.emplace("gops", std::ceil(all_qos["workload"] / 1000000000.0));
                merged_qos_.emplace("egops", std::ceil(all_qos["workload_on_arch"] / 1000000000.0));
#ifdef _WIN32
                TL_TRACE("gops from xmodel") << all_qos["workload"] / 1000000000.0 << ", graph-runner: " << this << ", ctx_idx_ = " << ctx_idx_;
#endif
              }

              // Parse the QoS arguments and efficient mode arguments.
              // Add Efficient Mode parameter to QoS to pass through to XRT UMD.
              parse_qos_args_and_efficient_mode(attrs_);

              if (attrs_->has_attr("default_initialize") && attrs_->get_attr<bool>("default_initialize")) {
                  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                      << "graph-runner create with default_initialize";
                  default_initialize_attrs(subgraph, attrs_, merged_qos_, xrt_context_);
              }
              else { // HW context come from upper layer SW stack through attrs
                  if (!match_one_xclbin_xmodel(attrs_->get_attr<xrt::xclbin*>("xrt_xclbin"), subgraph))  // xclbin object
                  {
#ifdef _WIN32
                      TL_TRACE("the fingerprint of given xclbin object doesn't match subgraph's!");
#endif
                      throw std::runtime_error("the fingerprint of given xclbin object doesn't match subgraph's");
                  }
              }

              this->kernel_runners_.emplace_back(KernelRunnerFactory::create("DpuKernelRunner", engine_, subgraph, attrs_));
          }
      }

      // If we are running Fake DPU, use FakeDpuKernelRunner
      if(attrs_->has_attr("fake_dpu") && attrs_->get_attr<bool>("fake_dpu"))
      {
        if(subgraph->get_attr<std::string>("device") == "DPU")
          this->kernel_runners_.emplace_back(KernelRunnerFactory::create("FakeDpuKernelRunner", engine_, subgraph, attrs_));
      }
    }

    // This code only matters for unit testing KernelRunner
    // Most likely will remove soon
    if(subgraph->has_attr("kernel") && subgraph->get_attr<std::string>("kernel") == "TestKernelRunner")
      this->kernel_runners_.emplace_back(KernelRunnerFactory::create("TestKernelRunner", engine_, subgraph, attrs_));

  });

  if (kernel_runners_.empty()) {
#ifdef _WIN32
      TL_TRACE("Could not determine KernelRunner for any subgraphs!");
#endif
      throw std::runtime_error("Error: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " Could not determine KernelRunner for any subgraphs.");
  }
  for (auto &kernel_runner : kernel_runners_)
  {
    tensor_buffers_.emplace_back(kernel_runner->create_inputs());
  }

  tensor_buffers_.emplace_back(kernel_runners_.back()->create_outputs());
  for (auto &v : tensor_buffers_)
  {
    tensor_buffers_raw_.emplace_back();
    std::transform(v.begin(), v.end(), std::back_inserter(tensor_buffers_raw_.back()), [](std::unique_ptr<vart::TensorBuffer> &sp)
                   { return sp.get(); });
  }
}

GraphRunner::~GraphRunner()
{


#ifdef _WIN32
    TL_TRACE_BLOCK("GraphRunner::~GraphRunner") << this;
#endif
  try {
    // xrt::bo objects must be destroyed before xrt::hw_context, because
    // BO teardown in driver requires access to valid hw_context data.
    kernel_runners_.clear();
    tensor_buffers_.clear();

    if (context_inner_) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "graph-runner release ctx_idx_ : " << ctx_idx_;
        xrt_context_.dec_ref_cnt(ctx_idx_, merged_qos_);
    }
    if (attrs_inner_)
        delete attrs_;
  } catch (const std::runtime_error& e) {
      std::cerr <<"runtime_error occured : " << e.what() << "\n";
  }	  
}

void GraphRunner::parse_qos_args_and_efficient_mode(xir::Attrs* attrs)
{
    // Get PerformancePreference and QoS from user's attr
    if (attrs_->has_attr("performance_preference") && attrs_->has_attr("qos_params")) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "graph-runner create with attrs include performance preference and QoS";

        std::string perf_pref = attrs_->get_attr<std::string>("performance_preference");
        auto qos = attrs_->get_attr<std::map<std::string, std::uint32_t>>("qos_params");

        // Check if priority is set and if there's any error conditions
        int32_t priority = -1;
        for (auto& q: qos) {
            priority = -1;
            if (q.first == "priority") {
                priority = q.second;

                // priority = realtime and PerformancePrefence = HighEfficiencyMode
                if (static_cast<uint32_t>(GraphEngine::qos_priority::qos_priority_realtime) == priority && 0 == perf_pref.compare("HighEfficiencyMode")) {
                    throw std::runtime_error("Realtime priority cannot mix with HighEfficiencyMode");
                }
            }

#ifdef _WIN32
            if (q.first != "fps" && q.first != "latency" && q.first != "priority") {
                TL_TRACE("QoS from User[Warning]") << q.first << " ==> " << q.second;
            }
            else {
                TL_TRACE("QoS from User") << q.first << " ==> " << q.second;
            }
#endif

            merged_qos_.insert(q);
        }
        if (0 == perf_pref.compare("HighEfficiencyMode"))
        {
            merged_qos_.emplace("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_high_efficiency));
        }
        else if (0 == perf_pref.compare("Default"))
        {
            merged_qos_.emplace("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_default));
        }
    }

    // Get only QoS from user's attr, PerformancePreference not set
    else if (attrs_->has_attr("qos_params")) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "graph-runner create with attrs include QoS only";
        auto qos = attrs_->get_attr<std::map<std::string, std::uint32_t>>("qos_params");

        for (auto& q : qos) {
#ifdef _WIN32
            if (q.first != "fps" && q.first != "latency" && q.first != "priority") {
                TL_TRACE("QoS from User[Warning]") << q.first << " ==> " << q.second;
            }
            else {
                TL_TRACE("QoS from User") << q.first << " ==> " << q.second;
            }
#endif
            merged_qos_.insert(q);
        }
    }

    // Get only PerformancePreference from user's attr, QoS not set
    else if (attrs_->has_attr("performance_preference")) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "graph-runner create with attrs include performance preference only";
        auto perf_pref = attrs_->get_attr<std::string>("performance_preference");

        if (0 == perf_pref.compare("HighEfficiencyMode"))
        {
            merged_qos_.emplace("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_high_efficiency));
        }
        else if (0 == perf_pref.compare("Default"))
        {
            merged_qos_.emplace("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_default));
        }
    }

    else {
#ifdef _WIN32
        TL_TRACE("No PerformancePreference or QoS given");
#endif
    }
}

std::pair<uint32_t, int> GraphRunner::execute_async(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs)
{
  uint32_t job_id = 0;
    if (async_exec_) {
        job_id = engine_->enqueue(
            [this, &inputs, &outputs] { this->run(inputs, outputs); });
#ifdef _WIN32
        TL_TRACE(__func__) << job_id << ", graph-runner: " << this << ", ctx_idx_ = " << ctx_idx_;
#endif
  } else {
    run(inputs, outputs);
  }
    return std::pair<uint32_t, int>(job_id, 0);
}

int GraphRunner::wait(int job_id, int timeout)
{
  if (async_exec_) {
    engine_->wait(job_id, timeout);
#ifdef _WIN32
    TL_TRACE(__func__) << job_id << ", graph-runner: " << this << ", ctx_idx_ = " << ctx_idx_;
#endif
  }
  return 0;
}

void GraphRunner::run(const std::vector<vart::TensorBuffer *> &inputs, const std::vector<vart::TensorBuffer *> &outputs)
{
#ifdef _WIN32
    TL_TRACE_BLOCK(" GraphRunner::run");
#endif
  for (int i = 0; i < kernel_runners_.size(); i++)
  {
    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
        << "graph-runner run idx: " << i;
    const std::vector<vart::TensorBuffer*> &source = (i == 0) ? inputs : tensor_buffers_raw_[i];
    const std::vector<vart::TensorBuffer*> &sink = (i == kernel_runners_.size()-1) ? outputs : tensor_buffers_raw_[i+1];
    kernel_runners_[i]->run(source, sink);
  }
}

std::pair<uint32_t, int> GraphRunner::execute(
    const std::vector<vart::TensorBuffer *> &inputs,
    const std::vector<vart::TensorBuffer *> &outputs, int timeout)
{
  auto job = execute_async(inputs, outputs);
  job.second = wait(job.first, timeout);
  return job;
}

std::vector<const xir::Tensor *> GraphRunner::get_input_tensors()
{
  std::vector<const xir::Tensor *> input_tensors;

  std::transform(tensor_buffers_raw_[0].cbegin(), tensor_buffers_raw_[0].cend(), std::back_inserter(input_tensors),
    [](const vart::TensorBuffer * tensor_buffer) {return tensor_buffer->get_tensor(); });

  return input_tensors;
}

std::vector<const xir::Tensor *> GraphRunner::get_output_tensors()
{
  std::vector<const xir::Tensor *> output_tensors;

  std::transform(tensor_buffers_raw_[std::size(tensor_buffers_raw_)-1].cbegin(), tensor_buffers_raw_[std::size(tensor_buffers_raw_)-1].cend(), std::back_inserter(output_tensors),
    [](const vart::TensorBuffer * tensor_buffer) {return tensor_buffer->get_tensor(); });

  return output_tensors;
}

std::vector<vart::TensorBuffer *> GraphRunner::get_inputs()
{
  return tensor_buffers_raw_[0];
}

std::vector<vart::TensorBuffer *> GraphRunner::get_outputs()
{
  return tensor_buffers_raw_[tensor_buffers_raw_.size() - 1];
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphRunner::create_inputs()
{
  return kernel_runners_[0]->create_inputs();
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphRunner::create_outputs()
{
  return kernel_runners_[kernel_runners_.size() - 1]->create_outputs();
}

std::vector<float> GraphRunner::get_input_scale_factors()
{
  std::vector<float> input_scale_factors;

  auto input_tensors = get_input_tensors();

  std::transform(input_tensors.cbegin(), input_tensors.cend(), std::back_inserter(input_scale_factors), get_input_scale);

  return input_scale_factors;
}

std::vector<float> GraphRunner::get_output_scale_factors()
{
  std::vector<float> output_scale_factors;

  auto output_tensors = get_output_tensors();

  std::transform(output_tensors.cbegin(), output_tensors.cend(), std::back_inserter(output_scale_factors), get_output_scale);

  return output_scale_factors;
}

int GraphRunner::set_run_attrs(std::unique_ptr<xir::Attrs>& attrs){
#ifdef _WIN32
    TL_TRACE("Called GraphRunner::set_run_attrs\n");
#endif
    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "Called GraphRunner::set_run_attrs";

    // Get PerformancePreference and QoS from user's attr
    if (attrs->has_attr("performance_preference") && !merged_qos_.empty()) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "graph-runner update with attrs include performance preference and QoS";
        std::string perf_pref = attrs->get_attr<std::string>("performance_preference");

        // Check if priority is set and if there's any error conditions
        int32_t priority = -1;
        for (auto& q: merged_qos_) {
            priority = -1;
            if (q.first == "priority") {
                priority = q.second;

                // priority = realtime and PerformancePreference = HighEfficiencyMode
                if (static_cast<uint32_t>(GraphEngine::qos_priority::qos_priority_realtime) == priority && 0 == perf_pref.compare("HighEfficiencyMode")) {
                    throw std::runtime_error("Realtime priority cannot mix with HighEfficiencyMode");
                }
            }
        }
#ifdef _WIN32
        TL_TRACE("Both performance preference and QoS given. Ignoring QoS, setting perf perf\n");
#endif
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "Both performance preference and QoS given. Ignoring QoS, setting perf perf";

        if (0 == perf_pref.compare("HighEfficiencyMode"))
        {
            merged_qos_.insert_or_assign("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_high_efficiency));
        }
        else if (0 == perf_pref.compare("Default"))
        {
            merged_qos_.insert_or_assign("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_default));
        }

        xrt_context_.GetContext(ctx_idx_).update_qos(merged_qos_);
    }

    // Get only QoS from user's attr, PerformancePreference not set
    else if (attrs->has_attr("qos_params")) {
#ifdef _WIN32
        TL_TRACE("Only QoS given, no performance preference. Not supported. Not action taken.\n");
#endif
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "Only QoS given, no performance preference. Not supported. Not action taken.";
    }

    // Get only PerformancePreference from user's attr, QoS not set
    else if (attrs->has_attr("performance_preference")) {
#ifdef _WIN32
        TL_TRACE("Only performance preference given. Setting perf pref.\n");
#endif
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "Only performance preference given. Setting perf pref.";
        auto perf_pref = attrs->get_attr<std::string>("performance_preference");

        if (0 == perf_pref.compare("HighEfficiencyMode"))
        {
            merged_qos_.insert_or_assign("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_high_efficiency));
        }
        else if (0 == perf_pref.compare("Default"))
        {
            merged_qos_.insert_or_assign("perf_pref", static_cast<uint32_t>(GraphEngine::performance_preference::perf_pref_default));
        }

        xrt_context_.GetContext(ctx_idx_).update_qos(merged_qos_);
    }

    else {
#ifdef _WIN32
        TL_TRACE("No PerformancePreference or QoS given");
#endif
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << "No PerformancePreference or QoS given";
    }

    return 0;
}

// Static member of GraphRunner Class
// All instances of said class will share this same object
// It must be initialized in the implementation file
Engine* GraphRunner::engine_ = get_engine_instance();

// Public APIs
std::unique_ptr<vart::RunnerExt> GraphEngine::create_graph_runner(const std::string &xmodel, xir::Attrs* attrs, bool bypass_pad) {
  return std::make_unique<GraphRunner>(xmodel, attrs, bypass_pad);
}

std::unique_ptr<vart::RunnerExt> GraphEngine::create_graph_runner(const xir::Graph *graph, xir::Attrs* attrs, bool bypass_pad) {
  return std::make_unique<GraphRunner>(graph, attrs, bypass_pad);
}

std::unique_ptr<vart::RunnerExt> GraphEngine::create_graph_runner(const xir::Subgraph *root, xir::Attrs* attrs, bool bypass_pad) {
  return std::make_unique<GraphRunner>(root, attrs, bypass_pad);
}

std::unique_ptr<vart::Runner> GraphEngine::create_dpu_runner(const std::string &xmodel, xir::Attrs* attrs) {

 // Read in XMODEL
 std::unique_ptr<xir::Graph> graph = xir::Graph::deserialize(xmodel);

  return GraphEngine::create_dpu_runner(graph.release(), attrs);
}

std::unique_ptr<vart::Runner> GraphEngine::create_dpu_runner(const xir::Graph *graph, xir::Attrs* attrs) {

  // Get the First FPGA Subgraph
  std::vector<const xir::Subgraph *> subgraphs = graph->get_root_subgraph()->children_topological_sort();
  auto subgraph = std::find_if(subgraphs.begin(), subgraphs.end(), [](const xir::Subgraph *sg) {
    return sg->get_attr<std::string>("device") == "DPU";
  });
  if (subgraph != subgraphs.end()) {
    return GraphEngine::create_dpu_runner(*subgraph, attrs);
  } else {
	return std::unique_ptr<vart::Runner>{};  
  } 
}

std::unique_ptr<vart::Runner> GraphEngine::create_dpu_runner(const xir::Subgraph *subgraph, xir::Attrs* attrs) {

  ASSERT(subgraph->has_attr("device"), "Can't create DPU Runner from subgraph that does not target DPU.");
  ASSERT(subgraph->get_attr<std::string>("device") == "DPU", "Can't create DPU Runner from subgraph that does not target DPU.");

  KernelRunnerFactory::register_kernel_runners();

  auto runner = KernelRunnerFactory::create("DpuKernelRunner", get_engine_instance(), subgraph, attrs);
  return runner;
}

static vart::Runner *create_runner_with_attrs_imp(const xir::Subgraph *subgraph,
                                                  xir::Attrs *attrs) {

  vitis::ai::trace::add_subgraph(subgraph);
  if (subgraph->has_attr("device") &&
      subgraph->get_attr<std::string>("device") == "DPU") {
    return new GraphRunner((xir::Subgraph *)subgraph, attrs);
  }
  return nullptr;
}

static vart::Runner* create_runner_imp(const xir::Subgraph* subgraph, const std::string& mode) {

  if(subgraph->has_attr("device") && subgraph->get_attr<std::string>("device") == "DPU") {
    KernelRunnerFactory::register_kernel_runners();
    auto runner = KernelRunnerFactory::create("DpuKernelRunner", get_engine_instance(), subgraph, nullptr);
    return runner.release();
  }

  return new GraphRunner(subgraph);
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_inputs(std::unique_ptr<vart::RunnerExt> &runner)
{
  return GraphEngine::create_inputs(runner.get());
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_inputs(vart::RunnerExt *runner)
{
  auto *graphRunner = dynamic_cast<GraphRunner*>(runner);
  return graphRunner ? graphRunner->create_inputs() : std::vector<std::unique_ptr<vart::TensorBuffer>>{};
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_outputs(std::unique_ptr<vart::RunnerExt> &runner)
{
  return GraphEngine::create_outputs(runner.get());
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_outputs(vart::RunnerExt *runner)
{
  auto *graphRunner = dynamic_cast<GraphRunner*>(runner);
  return graphRunner ? graphRunner->create_outputs(): std::vector<std::unique_ptr<vart::TensorBuffer>>{};
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_inputs(std::unique_ptr<vart::Runner> &runner)
{
  return GraphEngine::create_inputs(runner.get());
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_inputs(vart::Runner *runner)
{
  auto *dpuRunner = dynamic_cast<KernelRunner*>(runner);
  return dpuRunner->create_inputs();
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_outputs(std::unique_ptr<vart::Runner> &runner)
{
  return GraphEngine::create_outputs(runner.get());
}

std::vector<std::unique_ptr<vart::TensorBuffer>> GraphEngine::create_outputs(vart::Runner *runner)
{
  auto *dpuRunner = dynamic_cast<KernelRunner*>(runner);
  return dpuRunner->create_outputs();
}

#if GRAPH_ENGINE_USE_DLL == 1
extern "C" vart::Runner *create_runner_with_attrs(const xir::Subgraph *subgraph,
                                                  xir::Attrs *attrs) {
    return create_runner_with_attrs_imp(subgraph, attrs);
}
extern "C" vart::Runner* create_runner(const xir::Subgraph* subgraph, const std::string& mode) {
    return create_runner_imp(subgraph,mode);
}
#else
#include <vitis/ai/plugin.hpp>
namespace {
static class Register {
public:
  Register() {
    vitis::ai::register_plugin("libgraph-engine.so",
                               "create_runner_with_attrs",
                               (void*)&create_runner_with_attrs_imp);
    vitis::ai::register_plugin("graph-engine.dll",
                               "create_runner_with_attrs",
                               (void*)&create_runner_with_attrs_imp);
    vitis::ai::register_plugin("libgraph-engine.so",
                               "create_runner",
                               (void*)&create_runner_imp);
    vitis::ai::register_plugin("graph-engine.dll",
                               "create_runner",
                               (void*)&create_runner_imp);
  }
} __register;
} // namespace
extern "C" {
void* graph_engine__hook = &__register;
}
#endif
std::vector<float> GraphEngine::get_input_scale_factors(std::unique_ptr<vart::RunnerExt> &runner)
{
  return GraphEngine::get_input_scale_factors(runner.get());
}

std::vector<float> GraphEngine::get_input_scale_factors(vart::RunnerExt *runner)
{
  auto *graphRunner = dynamic_cast<GraphRunner*>(runner);
  return graphRunner ? graphRunner->get_input_scale_factors() : std::vector<float>{};
}

std::vector<float> GraphEngine::get_output_scale_factors(std::unique_ptr<vart::RunnerExt> &runner)
{
  return GraphEngine::get_output_scale_factors(runner.get());
}

std::vector<float> GraphEngine::get_output_scale_factors(vart::RunnerExt *runner)
{
  auto *graphRunner = dynamic_cast<GraphRunner*>(runner);
  return graphRunner ? graphRunner->get_output_scale_factors() : std::vector<float>{};
}

void GraphEngine::copy_buffer(vart::TensorBuffer *tb, std::vector<std::int8_t> &dataSrc)
{
  GraphEngine::copy_buffer(tb, dataSrc.data(), dataSrc.size());
}

void GraphEngine::copy_buffer(vart::TensorBuffer *tb, std::int8_t *dataSrc, size_t srcElements)
{
  #ifdef _WIN32
  TL_TRACE_BLOCK(__func__);
  #endif
  auto tensor = tb->get_tensor();
  auto dstElements = tensor->get_data_size();
  auto dataType = tensor->get_data_type();
  auto *dataDst = (std::int8_t*)(tb->data().first);
  ASSERT(dstElements == srcElements, "Size mismatch between source and destination buffers in copy operation.");

  ASSERT(dataType.type == xir::DataType::XINT, "Tensor Buffer has illegal dataType in copy operation.");

  ASSERT(dataDst != nullptr, "TensorBuffer memory is not allocated in buffer copy operation..");

  ASSERT(dataSrc != nullptr, "Source memory is not allocated in buffer copy operation.");

  std::memcpy(dataDst, dataSrc, dstElements);

}

void GraphEngine::copy_buffer(vart::TensorBuffer *tb, std::vector<float> &dataSrc)
{
  GraphEngine::copy_buffer(tb, dataSrc.data(), dataSrc.size());
}

void GraphEngine::copy_buffer(vart::TensorBuffer *tb, float *dataSrc, size_t srcElements)
{
  #ifdef _WIN32
  TL_TRACE_BLOCK(__func__);
  #endif
  auto tensor = tb->get_tensor();
  auto dstElements = tensor->get_element_num();
  auto dataType = tensor->get_data_type();
  auto dataPtr = (void*)(tb->data().first);


  ASSERT(dstElements == srcElements, "Size mismatch between source and destination buffers in copy operation.");

  ASSERT(dataType.type == xir::DataType::XINT || dataType.type == xir::DataType::INT || dataType.type == xir::DataType::FLOAT, "Tensor Buffer has illegal dataType in copy operation.");

  ASSERT(dataPtr != nullptr, "TensorBuffer memory is not allocated in buffer copy operation..");

  ASSERT(dataSrc != nullptr, "Source memory is not allocated in buffer copy operation.");

  if(dataType.type == xir::DataType::FLOAT)
  {
    auto dataDst = (float*)dataPtr;
    std::memcpy(dataDst, dataSrc, sizeof(float)*dstElements);
  // Need to scale float to int
  } 
  else if((dataType.type == xir::DataType::XINT || dataType.type == xir::DataType::INT) && dataType.bit_width == 8)
  {
    
    if(tensor->has_attr("fix_point"))
    {
    auto scale = get_input_scale(tensor);
    auto dataDst = (std::int8_t*)dataPtr;

    __TIC__(FLOAT_TO_FIX)
    float2fix(dataSrc, dataDst, srcElements, scale);
    __TOC__(FLOAT_TO_FIX)
    }
    else
    {
    auto [scale, zero_point] = get_input_scale_zero_point(tensor);
    auto dataDst = (std::int8_t*)dataPtr;

    __TIC__(FLOAT_TO_INT8)
    float2fix(dataSrc, dataDst, srcElements, scale, zero_point);
    __TOC__(FLOAT_TO_INT8)
    }
  } 
  else if(dataType.type == xir::DataType::INT && dataType.bit_width == 16)
  {
    auto [scale, zero_point] = get_input_scale_zero_point(tensor);
    auto dataDst = (std::int16_t*)dataPtr;

    __TIC__(FLOAT_TO_INT16)
    float2int16(dataSrc, dataDst, srcElements, scale, zero_point);
    __TOC__(FLOAT_TO_INT16)
  }
  else 
  {
    throw std::invalid_argument("Unsupported type: " + dataType.to_string());
  }
}

void GraphEngine::copy_buffer(std::vector<std::int8_t> &dataDst, vart::TensorBuffer *tb)
{
  GraphEngine::copy_buffer(dataDst.data(), dataDst.size(), tb);
}

void GraphEngine::copy_buffer(std::int8_t *dataDst, size_t dstElements, vart::TensorBuffer *tb)
{
  #ifdef _WIN32
  TL_TRACE_BLOCK(__func__);
  #endif
  auto tensor = tb->get_tensor();
  auto srcElements = tensor->get_data_size();
  auto srcShapes = tensor->get_shape();
  auto srcStrides = XrtBuffer::get_strides(tb); // I don't like that I depend on XrtBuffer class
  auto dataType = tensor->get_data_type();
  auto *dataSrc = (std::int8_t*)(tb->data().first);

  ASSERT(dstElements == srcElements, "Size mismatch between source and destination buffers in copy operation.");

  ASSERT(dataType.type == xir::DataType::XINT, "Tensor Buffer has illegal dataType in copy operation.");

  ASSERT(dataSrc != nullptr, "TensorBuffer memory is not allocated in buffer copy operation..");

  ASSERT(dataDst != nullptr, "Destination memory is not allocated in buffer copy operation.");

  std::memcpy(dataDst, dataSrc, dstElements);
}

void GraphEngine::copy_buffer(std::vector<float> &dataDst, vart::TensorBuffer *tb)
{
  GraphEngine::copy_buffer(dataDst.data(), dataDst.size(), tb);
}

void GraphEngine::copy_buffer(float *dataDst, size_t dstElements, vart::TensorBuffer *tb)
{
  #ifdef _WIN32
  TL_TRACE_BLOCK(__func__);
  #endif
  auto tensor = tb->get_tensor();
  // auto srcElements = tensor->get_element_num();
  auto [dataPtr, srcBytes] = tb->data();
  auto dataType = tensor->get_data_type();
  auto srcElements = srcBytes / (dataType.bit_width / CHAR_BIT);

  ASSERT(dstElements == srcElements, "Size mismatch between source and destination buffers in copy operation.");

  ASSERT(dataType.type == xir::DataType::XINT || dataType.type == xir::DataType::INT || dataType.type == xir::DataType::FLOAT, "Tensor Buffer has illegal dataType in copy operation.");

  ASSERT((void*)dataPtr != nullptr, "TensorBuffer memory is not allocated in buffer copy operation..");

  ASSERT(dataDst != nullptr, "Destination memory is not allocated in buffer copy operation.");
  if(dataType.type == xir::DataType::FLOAT)
  {
    auto dataSrc = (float*)dataPtr;
    std::memcpy(dataDst, dataSrc, sizeof(float)*dstElements);
  }
  else if((dataType.type == xir::DataType::XINT || dataType.type == xir::DataType::INT) && dataType.bit_width == 8) // Need to scale int to float
  {
	    
    if(tensor->has_attr("fix_point"))
    {
    auto scale = get_output_scale(tensor);
    auto dataSrc = (std::int8_t*)dataPtr;

    __TIC__(FIX_TO_FLOAT)
    fix2float(dataSrc, dataDst, srcElements, scale);
    __TOC__(FIX_TO_FLOAT)
    }
    else
    {
    auto [scale, zero_point] = get_output_scale_zero_point(tensor);
    auto dataSrc = (std::int8_t*)dataPtr;

    __TIC__(INT8_TO_FLOAT)
    fix2float(dataSrc, dataDst, srcElements, scale, zero_point);
    __TOC__(INT8_TO_FLOAT)
    }
	    
  }
  else if(dataType.type == xir::DataType::INT && dataType.bit_width == 16) // Need to scale int16 to float
  {
    auto scale_zero_point = get_output_scale_zero_point(tensor);
    auto scale = scale_zero_point.first;
    auto zero_point = scale_zero_point.second;
    auto dataSrc = (std::int16_t*)dataPtr;

    __TIC__(INT16_TO_FLOAT)
    int162float(dataSrc, dataDst, srcElements, scale, zero_point);
    __TOC__(INT16_TO_FLOAT)
  } else {
    throw std::invalid_argument("Unsupported type: " + dataType.to_string());
  }
}

void GraphEngine::copy_buffer(vart::TensorBuffer *tb, int16_t *dataSrc, size_t srcElements)
{
  #ifdef _WIN32
  TL_TRACE_BLOCK(__func__);
  #endif
  auto tensor = tb->get_tensor();
  auto dstElements = tensor->get_element_num();
  auto dataType = tensor->get_data_type();
  auto dataPtr = (void*)(tb->data().first);

  ASSERT(dstElements == srcElements, "Size mismatch between source and destination buffers in copy operation.");

  ASSERT(dataType.type == xir::DataType::INT && dataType.bit_width == 16, "Tensor Buffer has illegal dataType in copy operation.");

  ASSERT(dataPtr != nullptr, "TensorBuffer memory is not allocated in buffer copy operation..");

  ASSERT(dataSrc != nullptr, "Source memory is not allocated in buffer copy operation.");

  if(dataType.type == xir::DataType::INT && dataType.bit_width == 16)
  {
    auto dataDst = (int16_t*)dataPtr;
    std::memcpy(dataDst, dataSrc, sizeof(int16_t)*dstElements);
  } else {
    throw std::invalid_argument("Unsupported type: " + dataType.to_string());
  }
}

void GraphEngine::copy_buffer(vart::TensorBuffer* tb, std::vector<int16_t>& dataSrc)
{
    GraphEngine::copy_buffer(tb, (int16_t*)dataSrc.data(), dataSrc.size());
}

void GraphEngine::copy_buffer(std::vector<int16_t> &dataDst, vart::TensorBuffer *tb)
{
  GraphEngine::copy_buffer(dataDst.data(), dataDst.size(), tb);
}

void GraphEngine::copy_buffer(int16_t *dataDst, size_t dstElements, vart::TensorBuffer *tb)
{
  #ifdef _WIN32
  TL_TRACE_BLOCK(__func__);
  #endif
  auto tensor = tb->get_tensor();
  // auto srcElements = tensor->get_element_num();
  auto [dataPtr, srcBytes] = tb->data();
  auto dataType = tensor->get_data_type();
  auto srcElements = srcBytes / (dataType.bit_width / CHAR_BIT);

  ASSERT(dstElements == srcElements, "Size mismatch between source and destination buffers in copy operation.");

  ASSERT(dataType.type == xir::DataType::INT && dataType.bit_width == 16,"Tensor Buffer has illegal dataType in copy operation.");

  ASSERT((void*)dataPtr != nullptr, "TensorBuffer memory is not allocated in buffer copy operation..");

  ASSERT(dataDst != nullptr, "Destination memory is not allocated in buffer copy operation.");

  if(dataType.type == xir::DataType::INT && dataType.bit_width == 16)
  {
    auto dataSrc = (int16_t*)dataPtr;
    std::memcpy(dataDst, dataSrc, sizeof(int16_t)*dstElements);
  } else {
    throw std::invalid_argument("Unsupported type: " + dataType.to_string());
  }
}

int GraphEngine::set_run_attrs(std::unique_ptr<vart::RunnerExt>& runner, std::unique_ptr<xir::Attrs>& attr)
{
    return runner->set_run_attrs(attr);
}

extern "C" {
#ifdef _WIN32
    __declspec(dllexport) std::unique_ptr<vart::RunnerExt> create_graph_runner_cif(const std::string& xmodel, xir::Attrs* attrs, bool bypass_pad) {
        return GraphEngine::create_graph_runner(xmodel, attrs, bypass_pad);
    }
    __declspec(dllexport) std::unique_ptr<vart::RunnerExt> create_graph_runner_graph_cif(const xir::Graph* graph, xir::Attrs* attrs, bool bypass_pad) {
        return GraphEngine::create_graph_runner(graph, attrs, bypass_pad);
    }
    __declspec(dllexport) std::unique_ptr<vart::RunnerExt> create_graph_runner_subgraph_cif(const xir::Subgraph* root, xir::Attrs* attrs, bool bypass_pad) {
        return GraphEngine::create_graph_runner(root, attrs, bypass_pad);
    }
    __declspec(dllexport) void copy_buffer_in_int8_cif(vart::TensorBuffer* tb, std::vector<std::int8_t>& dataSrc) {
        return GraphEngine::copy_buffer(tb, dataSrc);
    }
    __declspec(dllexport) void copy_buffer_in_float_cif(vart::TensorBuffer* tb, std::vector<float>& dataSrc) {
        return GraphEngine::copy_buffer(tb, dataSrc);
    }
    __declspec(dllexport) void copy_buffer_in_int16_cif(vart::TensorBuffer* tb, std::vector<int16_t>& dataSrc) {
        return GraphEngine::copy_buffer(tb, dataSrc);
    }
    __declspec(dllexport) void copy_buffer_out_int8_cif(std::vector<std::int8_t>& dataDst, vart::TensorBuffer* tb) {
        return GraphEngine::copy_buffer(dataDst, tb);
    }
    __declspec(dllexport) void copy_buffer_out_float_cif(std::vector<float>& dataDst, vart::TensorBuffer* tb) {
        return GraphEngine::copy_buffer(dataDst, tb);
    }
    __declspec(dllexport) void copy_buffer_out_int16_cif(std::vector<int16_t>& dataDst, vart::TensorBuffer* tb) {
        return GraphEngine::copy_buffer(dataDst, tb);
    }
#else
    std::unique_ptr<vart::RunnerExt> create_graph_runner_cif(const std::string& xmodel, xir::Attrs* attrs, bool bypass_pad) {
        return GraphEngine::create_graph_runner(xmodel, attrs, bypass_pad);
    }
    std::unique_ptr<vart::RunnerExt> create_graph_runner_graph_cif(const xir::Graph* graph, xir::Attrs* attrs, bool bypass_pad) {
        return GraphEngine::create_graph_runner(graph, attrs, bypass_pad);
    }
    std::unique_ptr<vart::RunnerExt> create_graph_runner_subgraph_cif(const xir::Subgraph* root, xir::Attrs* attrs, bool bypass_pad) {
        return GraphEngine::create_graph_runner(root, attrs, bypass_pad);
    }
    void copy_buffer_in_int8_cif(vart::TensorBuffer* tb, std::vector<std::int8_t>& dataSrc) {
        return GraphEngine::copy_buffer(tb, dataSrc);
    }
    void copy_buffer_in_float_cif(vart::TensorBuffer* tb, std::vector<float>& dataSrc) {
        return GraphEngine::copy_buffer(tb, dataSrc);
    }
    void copy_buffer_in_int16_cif(vart::TensorBuffer* tb, std::vector<int16_t>& dataSrc) {
        return GraphEngine::copy_buffer(tb, dataSrc);
    }
    void copy_buffer_out_int8_cif(std::vector<std::int8_t>& dataDst, vart::TensorBuffer* tb) {
        return GraphEngine::copy_buffer(dataDst, tb);
    }
    void copy_buffer_out_float_cif(std::vector<float>& dataDst, vart::TensorBuffer* tb) {
        return GraphEngine::copy_buffer(dataDst, tb);
    }
    void copy_buffer_out_int16_cif(std::vector<int16_t>& dataDst, vart::TensorBuffer* tb) {
        return GraphEngine::copy_buffer(dataDst, tb);
    }
#endif
}
