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

#include <numeric>
#include <memory>
#include <assert.h>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <unordered_map>

#include "graph-engine/kernel_runner.hpp"
#include "graph-engine/graph_runner.hpp"
#include "graph-engine/patch_ddr_address.hpp"

#include <xrt/xrt_device.h>
#include <experimental/xrt_hw_context.h>
#include <experimental/xrt_ext.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include "utils/FastPad.hpp"
#include "utils/FastDepad.hpp"
#include "vitis/ai/profiling.hpp"

#ifdef VART_IN_BUILD_TREE
#include <vitis/ai/trace.hpp>
#else
#include <vart/trace/trace.hpp>
#endif

#ifdef _WIN32
#include <vitis/ai/tracelogging.hpp>
#endif

DEF_ENV_PARAM(XLNX_XRT_CU_DRY_RUN, "0");

namespace fs = std::filesystem;
using address_info_t = std::tuple<uint64_t, xir::Tensor*, std::string, int32_t, int32_t, std::string>;

/**
 * @class DpuKernelRunner
 *
 * @brief
 * DpuKernelRunner is a concrete KernelRunner class that implements a simple test kernel.
 *
 * @details
 * All KernelRunners must be constructed with two arguments:
 *  - A pointer to a TaskPool "engine"
 *  - A pointer to an xir::subgraph
 * A KernelRunner must provide a run method that defines the transfer funtion
 * of how to process a batch of inputs and generate data to a batch of outputs.
 *
 * The base class also provides common functionality such as extracting input and output shapes.
 */
class DpuKernelRunner : public KernelRunner
{
    struct TensorInfo
    {
        bool padNeeded = false;
        size_t element_stride;
        xrt::bo paddedSubBo;
        int32_t padOffset;
        std::array<std::int32_t, 4> paddedOutputShape;
        std::array<std::pair<std::int32_t, std::int32_t>, 4> dstSlices;
        std::vector<std::int32_t> paddings;
    };

    struct DepadTensorInfo
    {
        DepadTensorInfo(xrt::bo& bo, const std::vector<int32_t> &shapes, const std::vector<int32_t> &strides, std::int32_t ddr_offset)
            : shapes{shapes}
            , strides{strides}
            , ddr_offset{ddr_offset}
            , output_bo{bo}
        {}
        xrt::bo output_bo;
        std::vector<int32_t> shapes;
        std::vector<int32_t> strides;
        std::int32_t ddr_offset;
    };

public:
    // This enum defines the names of the xrt::kernel arguments, and creates a name -> position mapping
    enum
    {
        MODE_ARG,
        INPUT_ARG,
        PARAMETERS_ARG,
        OUTPUT_ARG,
        INTERMEDIATE_ARG,
        INSTRUCTIONS_ARG,
        INSTRUCTIONS_SIZE_ARG,
        BO_MC_ARG
    };
    void super_layer_init(int pdi_iter, const xir::Subgraph* subgraph, 
        const xir::Subgraph* child_iter, xrt::hw_context& hw_context,
        xrt::kernel& kernel,  int offset, int32_t input_layer_index,
        int output_layer_index) {
        auto child_name = child_iter->get_name();
        auto child = child_iter;
        layer_info layer(child_name);
        if (child->has_attr("workload")) {
            layer.set_workload(child->get_attr<uint64_t>("workload"));
        }
        layer.set_depth(child->get_depth());
        xrt::bo sub_bo;

        if (child->has_attr("mc_code")) {
            auto& mc_code = child->get_attr<std::vector<char> >("mc_code");
            if (mc_code.size() == 0)
                return;
            if (attrs_->has_attr("bo_sram")) {
                uint32_t alignment = 32 << 10;
                int align_bo_size = (((int)mc_code.size() + (alignment - 1)) / alignment) * alignment;
                sub_bo = xrt::bo(instructions_, align_bo_size, offset);
                offset += align_bo_size;
            }
            else {
                sub_bo = xrt::bo(hw_context, mc_code.size(), xrt::bo::flags::cacheable, kernel.group_id(INSTRUCTIONS_ARG));
            }
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << " DPU runner superlayer subgraph info: " << child_name
                << ",  sub bo: " << sub_bo.address()
                << ",  size: " << mc_code.size();
            sub_bo.write(mc_code.data(), mc_code.size(), 0);
            sub_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            std::get<0>(layer.code_addr) = sub_bo;
            std::get<1>(layer.code_addr) = sub_bo.size() / sizeof(int);
        }
        auto in_tensors = child->get_input_tensors();
        for (auto t : in_tensors) {
            auto tensor = find_tensor(t, subgraph, true);
            int reg_id = 0xff;
            if (tensor->has_attr("reg_id"))
                reg_id = tensor->get_attr<int32_t>("reg_id");
            address_info_t in = std::make_tuple(tensor->get_attr<int32_t>("ddr_addr"),
                (xir::Tensor*)t, layer_info::name_map(t->get_name()), reg_id, input_layer_index, "");
            input_layer_index++;
            layer.inputs.emplace_back(in);
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << " DPU runner superlayer subgraph input tensor: " << t->get_name()
                << ",  reg_id: " << reg_id
                << ",  ddr_addr: " << tensor->get_attr<int32_t>("ddr_addr");

        }
        auto out_tensors = child->get_output_tensors();
        std::string fusion_memory_status = "";
        if (child->has_attr("fusion_memory_status")) {
            fusion_memory_status = child->get_attr<std::vector<std::string>>("fusion_memory_status")[0];
        }
        for (auto t : out_tensors) {
            auto tensor = find_tensor(t, subgraph, false);
            int reg_id = 0xff;
            if (tensor->has_attr("reg_id"))
                reg_id = tensor->get_attr<int32_t>("reg_id");
            layer.outputs.emplace_back(address_info_t(tensor->get_attr<int32_t>("ddr_addr"),
                (xir::Tensor*)t, layer_info::name_map(t->get_name()), reg_id, output_layer_index, fusion_memory_status));
            output_layer_index++;
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << " DPU runner superlayer subgraph output tensor: " << t->get_name()
                << ",  reg_id: " << reg_id
                << ",  ddr_addr: " << tensor->get_attr<int32_t>("ddr_addr");

        }
        dbg_layers_[pdi_iter].emplace_back(std::move(layer));
    }
    /**
     * Constructor for concrete KernelRunner.
     *
     * @param engine
     *  This is a pointer to a global task pool used for submitting execution requests.
     * @param subgraph
     *  This is a pointer to a subgraph. A subgraph is an object that aggregates
     *    the necessary meta data needed by a KernelRunner.
     *
     * This is an example of how to define your own KernelRunner.
     * It is also used for unit testing.
     */
    DpuKernelRunner(Engine* engine, const xir::Subgraph* subgraph, xir::Attrs* attrs)
        : KernelRunner(engine, subgraph)
    {
        if (!attrs || !attrs->has_attr("xrt_device") || !attrs->has_attr("xrt_xclbin") || !attrs->has_attr("xrt_hw_context"))
        {
#ifdef _WIN32
            TL_TRACE("DpuKernelRunner::DpuKernelRunner throw, attrs = ") << attrs;
#endif
            throw std::runtime_error("DpuKernelRunner cannot access metadata from attrs.");
        }

        // Extract XRT objects from xir::Attrs
        xrt::device& device = *(attrs->get_attr<xrt::device*>("xrt_device"));
        xrt::xclbin& xclbin = *(attrs->get_attr<xrt::xclbin*>("xrt_xclbin"));
        xrt::hw_context& hw_context = *(attrs->get_attr<xrt::hw_context*>("xrt_hw_context"));
        // Save attrs
        attrs_ = attrs;

        int total_size = 0;
        xrt::kernel kernel;
        std::string kernel_name;
        debug_mode_ = (bool)stoi(get_env_var("XLNX_ENABLE_DEBUG_MODE", "0"));
        dump_mode_ = (bool)stoi(get_env_var("XLNX_ENABLE_DUMP", "0"));
        //# Check for PDI enable
        bool en_pdi = is_pdi_enabled(subgraph);

        if (en_pdi) {
            bool find = false;
            auto pdi_subgraphs = subgraph->children_topological_sort();
            std::for_each(pdi_subgraphs.cbegin(), pdi_subgraphs.cend(), [&](const xir::Subgraph* pdi_subgraph) {
                if ((pdi_subgraph->get_attr<std::string>("type") == "PDI")) {
                    find = true;
                    // Extract Model Instructions From Subgraph
                    std::vector<char> pdi_mc_code = pdi_subgraph->get_attr<std::vector<char> >("mc_code");
                    if ((int)pdi_mc_code.size()) {
                        uint32_t words = (int)pdi_mc_code.size() / sizeof(int);
                        std::string attr_name = pdi_subgraph->get_name() + '_' + pdi_subgraph->get_attr<std::string>("name");
                        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                            << "DPU runner create pdi initialize: " << attr_name;
                        if (!attrs->has_attr(attr_name)) {
#ifdef _WIN32
                            TL_TRACE("DpuKernelRunner::DpuKernelRunner throw, attrs = ") << attrs;
#endif
                            throw std::runtime_error("DpuKernelRunner cannot access kernel metadata from attrs: " + attr_name);
                        }
                        kernel = *(attrs->get_attr<xrt::kernel*>(attr_name));
                        run_vec_.push_back(xrt::run(kernel));
                        //# Write to vector
                        instructionsVec_.push_back(pdi_mc_code);
                        instructionWords_.push_back(words);
                        //32kb alignment
                        uint32_t alignment = 32 << 10;
                        total_size += (((int)pdi_mc_code.size() + (alignment - 1)) / alignment) * alignment;
                        kernel_name = pdi_subgraph->get_attr<std::string>("name");
                    }
                }
            });
            if (!find)
                throw std::runtime_error("Couldn't find PDI subgraph that contains mc_code ");
        }
        else {
            if (!attrs->has_attr("xrt_kernel")) {
#ifdef _WIN32
                TL_TRACE("DpuKernelRunner::DpuKernelRunner throw, attrs = ") << attrs;
#endif
                throw std::runtime_error("DpuKernelRunner cannot access kernel metadata from attrs: xrt_kernel");
            }
            kernel_name = get_xclbin_kernelName(xclbin);
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << "DPU runner create initialize";
            // Extract Model Instructions From Subgraph and Load int XRT BO
            instructionsVec_.push_back(subgraph->get_attr<std::vector<char> >("mc_code"));
            total_size = (int)instructionsVec_[0].size();
            instructionWords_.push_back((int)instructionsVec_[0].size() / sizeof(int));
            kernel = xrt::kernel(hw_context, kernel_name);
            run_vec_.push_back(xrt::run(kernel));
        }

#ifdef _WIN32
        TL_TRACE("Total number of PM swaps: ") << run_vec_.size() - 1;
#endif


        XrtContext& xrtContext = XrtContext::GetInstance();
        sg_name_ = layer_info::name_map(subgraph->get_name());
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << sg_name_ << " DPU runner debug mode: " << debug_mode_ << ",  dump mode: " << dump_mode_;

        if (dump_mode_) {
            auto dump_folder_env = get_env_var("DEBUG_DUMP_DIR", "0");
            if (dump_folder_env != "0") {
                dump_folder_ = fs::path(dump_folder_env) / sg_name_;
            }
            else {
#ifdef _WIN32
                dump_folder_ = fs::path("C:\\dump") / sg_name_;
#else
                dump_folder_ = fs::path("/tmp") / "dump" / sg_name_;
#endif
            }
            if (!fs::exists(dump_folder_) && !fs::create_directories(dump_folder_)) {
#ifdef _WIN32
                TL_TRACE("DpuKernelRunner::DpuKernelRunner throw, create dump folder error");
#endif
                throw std::runtime_error("create dump folder error");
            }
        }

        std::vector<char> boMcVec;

        std::string regIdAddrPatch = getAddrPatchReg(subgraph);

        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << " DPU runner regIdAddrPatch: " << regIdAddrPatch;
        // Extract Model Parameters From Subgraph and Load int XRT BO
        if (subgraph->has_attr("reg_id_to_parameter_value")) {
            auto parametersMap = subgraph->get_attr<std::map<std::string, std::vector<char>>>("reg_id_to_parameter_value");
            std::vector<char>& parametersVec = parametersMap["REG_0"];
            if (parametersVec.size()) {
                parameters_ = xrt::bo(hw_context, parametersVec.size(), xrt::bo::flags::host_only, kernel.group_id(PARAMETERS_ARG));

                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner parameter size: " << parametersVec.size() << ",  bo address: " << parameters_.address();
                parameters_.write(parametersVec.data());
                parameters_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
            else {
                parameters_ = xrt::bo(hw_context, 4, xrt::bo::flags::host_only, kernel.group_id(PARAMETERS_ARG));
            }
            boMcVec = parametersMap[regIdAddrPatch];
            if (boMcVec.size() == 0) {
                regIdAddrPatch = "";
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner shim dma buffer size = 0 ";
            }
        }
        else {
            parameters_ = xrt::bo(hw_context, 4, xrt::bo::flags::host_only, kernel.group_id(PARAMETERS_ARG));
        }
        const bool addr_patch_needed = (regIdAddrPatch != "");

        args_num_ = (int)xclbin.get_kernel(kernel_name).get_num_args();
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << " DPU runner args num = " << args_num_;

        if (addr_patch_needed)
        {
            //boMcVec = parametersMap[regIdAddrPatch];
            bo_mc_ = xrt::bo(hw_context, boMcVec.size(), xrt::bo::flags::host_only, kernel.group_id(BO_MC_ARG));
            bo_mc_.write(boMcVec.data());
        }
        else if (args_num_ > BO_MC_ARG) // there is no AddrPatchNeeded, still needs a dummy buffer to match the args_num_ in xclbin
        {
            bo_mc_ = xrt::bo(hw_context, DUMMY_MC_CODE_BUFFER_SIZE, xrt::bo::flags::host_only, kernel.group_id(BO_MC_ARG));
        }

        padNeeded_ = false;
        padBufferSize_ = 0;
        if ((!attrs->has_attr("bypass_pad")) || (attrs->has_attr("bypass_pad") && (attrs->get_attr<bool>("bypass_pad") == false))) {
            for (auto& input_tensor : input_tensors_) {
                const auto &tensorName = input_tensor->get_name();
                size_t element_stride = input_tensor->get_data_type().bit_width / 8;
                input_tensor_info_map_[tensorName].element_stride = element_stride;

                std::vector<int32_t> unpadShape = input_tensor->get_shape(); // for implicit padding, upload op's input and output tensor shapes should be equal to the unpadded tensor's shape

                if (!input_tensor->has_attr("stride")) {
#ifdef _WIN32
                    TL_TRACE("DpuKernelRunner::DpuKernelRunner throw, input tensor has no stride : ") << tensorName;
#endif
                    throw std::runtime_error("No input stride exists for this tensor: " + tensorName);
                }

                std::vector<int32_t> strides = input_tensor->get_attr<std::vector<std::int32_t>>("stride");
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " tensor: " << tensorName
                    << ",  num: " << input_tensor->get_element_num() / unpadShape[0]
                    << ",  strides[0]: " << strides[0];

                // It is equivalent to ignoring the first dimension, so that we can support future models in which the first dimension is not 1.
                // The current version of the compiler and the vaip team agree that shape[0] must be 1
                if (strides[0] > input_tensor->get_element_num() / unpadShape[0]) {
                    padNeeded_ = true;
                    input_tensor_info_map_[tensorName].padNeeded = true;
                    input_tensor_info_map_[tensorName].padOffset = padBufferSize_;
                    padBufferSize_ += strides[0]* element_stride * unpadShape[0];
                } else {
                    input_tensor_info_map_[tensorName].padNeeded = false;
                }

                std::vector<int> paddings(unpadShape.size() * 2, 0);
                for (int i = 1; i < unpadShape.size(); i++) {
                    paddings[i * 2] = 0;
                    paddings[i * 2 + 1] = (strides[i - 1] / strides[i]) - unpadShape[i];
                }
                input_tensor_info_map_[tensorName].paddings = paddings;
            }

        } else {
            if (attrs->has_attr("bypass_pad") && (attrs->get_attr<bool>("bypass_pad") == true))
            {
                padNeeded_ = false;
                for (auto& tensor : input_tensors_)
                    input_tensor_info_map_[tensor->get_name()].padNeeded = true;
            }
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << " DPU runner pad: " << padNeeded_;
        // Determine buffer sizes
        auto sizeMap = subgraph->get_attr<std::map<std::string, std::int32_t>>("reg_id_to_size");
        if (sizeMap.find("REG_1") != sizeMap.end())
            intermediateSize_ = sizeMap["REG_1"];

        inputSize_ = sizeMap["REG_2"];
        input_ = xrt::ext::bo(hw_context, inputSize_, xrt::ext::bo::access_mode::read);

        outputSize_ = sizeMap["REG_3"];
        // Create Intermediate buffer for DDR spill
        intermediate_ = xrt::bo(hw_context, (intermediateSize_ > 0) ? intermediateSize_ : 1024 * 1024, xrt::bo::flags::host_only, kernel.group_id(INTERMEDIATE_ARG));

        // Create Output buffer
        output_ = xrt::bo(hw_context, outputSize_, xrt::bo::flags::host_only, kernel.group_id(OUTPUT_ARG));

        for (int i = 0; i < input_tensors_.size(); i++) {
            auto& input_tensor = input_tensors_[i];
            const auto &tensorName = input_tensor->get_name();

            if (padNeeded_) {
                std::vector<int32_t> unpadShape = input_tensor->get_shape();
                auto element_stride = input_tensor_info_map_[tensorName].element_stride;
                if (unpadShape.size() > 4) {
#ifdef _WIN32
                    TL_TRACE("DpuKernelRunner::DpuKernelRunner throw, invalid tensor, dimension > 4 : ") << tensorName;
#endif
                    throw std::runtime_error("Unsupported shape dimension > 4: " + tensorName);
                }
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner input tensor(pad): " << tensorName << ",  dims: " << unpadShape.size();
                std::array<std::int32_t, 4> shape = { 1, 1, 1, 1 };
                std::vector<int32_t> unpadShape_std(4, 1);
                auto paddings = input_tensor_info_map_[tensorName].paddings;
                for (int i = 0; i < unpadShape.size(); i++) {
                    shape[i + 4 - unpadShape.size()] = paddings[i * 2] + unpadShape[i] + paddings[i * 2 + 1];
                    unpadShape_std[i + 4 - unpadShape.size()] = unpadShape[i];
                }
                shape[3] *= element_stride;
                input_tensor_info_map_[tensorName].paddedOutputShape = shape;
                int32_t paddedOutSize = shape[0] * shape[1] * shape[2] * shape[3];
                std::int32_t ddrAdr = input_tensor->get_attr<std::int32_t>("ddr_addr");
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner input tensor(pad): " << tensorName
                    << ",  addr_addr: " << ddrAdr
                    << ",  tensor dimension: " << unpadShape.size()
                    << ",  paddedOut size: " << paddedOutSize
                    << ",  paddedInput size " << input_.size()
                    << ",  reg_2 size "<< input_.size()
                    << ",  reg_3 size " << output_.size();
                switch (input_tensor->get_attr<int>("reg_id")){
                    case 2: {
                        input_tensor_info_map_[tensorName].paddedSubBo = xrt::bo(input_, paddedOutSize, ddrAdr);
                        break;
                    }
                    case 3: {
                        input_tensor_info_map_[tensorName].paddedSubBo = xrt::bo(output_, paddedOutSize, ddrAdr);
                        break;
                    }
                    default: {
                        throw std::runtime_error("Invalid reg id for this tensor: " + tensorName);
                    }
                }

                // Every dimension will have two parameters... before, after
                // Convert this into a vector of pairs, so we can iterate over dimensions easier
                std::vector<std::pair<int, int>> dimPaddings;
                for (unsigned int i = 0; i < unpadShape_std.size() - unpadShape.size(); i++) {
                    dimPaddings.emplace_back(0, unpadShape_std[i]);
                }
                for (unsigned int i = 0; i < paddings.size(); i += 2)
                    dimPaddings.emplace_back(paddings[i], paddings[i + 1]);
                // Compute index ranges in destination where data will need to be copied
                std::array<std::pair<std::int32_t, std::int32_t>, 4> dimspad;
                for (unsigned int i = 0; i < 4; i++)
                    dimspad[i] = std::make_pair(dimPaddings[i].first, dimPaddings[i].first + unpadShape_std[i]);
                dimspad[3].first *= element_stride;
                dimspad[3].second *= element_stride;
                input_tensor_info_map_[tensorName].dstSlices = dimspad;
            }
            else
            {
                if (!input_tensor->has_attr("ddr_addr"))
                    throw std::runtime_error("No input ddr_addr exists for this tensor: " + tensorName);

                if (!input_tensor->has_attr("stride"))
                    throw std::runtime_error("No input stride exists for this tensor: " + tensorName);
            }
        }

        // Create Input buffer
        if (padNeeded_)
            pad_input_ = xrt::bo(hw_context, padBufferSize_, xrt::bo::flags::host_only, kernel.group_id(INPUT_ARG));

        layer_info layer(subgraph->get_name());
        if (subgraph->has_attr("workload")) {
            layer.set_workload(subgraph->get_attr<uint64_t>("workload"));
        }

        // Annotate Tensors with information required by global memory allocator
        int32_t input_layer_index = 0;
        for (auto& input_tensor : input_tensors_)
        {
            input_tensor->set_attr<std::int32_t>("buffer_type", 2);
            input_tensor->set_attr<xrt::bo*>("parent_bo", &input_);
            int offset = input_tensor->get_attr<std::int32_t>("ddr_addr");
            address_info_t in = std::make_tuple(offset,
                input_tensor, layer_info::name_map(input_tensor->get_name()), input_tensor->get_attr<int>("reg_id"), input_layer_index, "");
            input_layer_index++;
            layer.inputs.emplace_back(in);
        }

        int32_t output_layer_index = 0;
        for (auto& output_tensor : output_tensors_)
        {
            output_tensor->set_attr<std::int32_t>("buffer_type", 2);
            output_tensor->set_attr<xrt::bo*>("parent_bo", &output_);
            layer.outputs.emplace_back(address_info_t(output_tensor->get_attr<std::int32_t>("ddr_addr"),
                output_tensor, layer_info::name_map(output_tensor->get_name()), output_tensor->get_attr<int>("reg_id"), output_layer_index, ""));
            output_layer_index++;
        }

        dbg_layers_.clear();
        dbg_layers_.resize(run_vec_.size());
        for (int i = 0; i < run_vec_.size(); i++)
            dbg_layers_[i].emplace_back(run_vec_.size() == 1 ? std::move(layer) : layer);
		
        // Extract Model Instructions From Subgraph and Load int XRT BO
        if (!debug_mode_) {

            if (attrs->has_attr("bo_sram")) {
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner create with bo_sram";

                instructions_ = *(attrs->get_attr<xrt::bo*>("bo_sram"));
                //If instructions_ is a slice then the following en_pdi loop will fail when we try to fill in pdi instructions TO-DO: follow up with xrt team to identify and error out if it's a slice
            }
            else {
                instructions_ = xrt::bo(hw_context, total_size, xrt::bo::flags::cacheable, kernel.group_id(INSTRUCTIONS_ARG));
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner instruction without default_initialize, bo address: "
                    << instructions_.address() << ",  bo size: " << total_size;

            }

            if (en_pdi) {
                size_t offset = 0;
                if (attrs->has_attr("bo_sram"))
                    if (attrs->has_attr("bo_offset")) offset = attrs->get_attr<size_t>("bo_offset");
                for (int i = 0; i < instructionsVec_.size(); i++) {
                    // 32KB alignment
                    uint32_t alignment = 32 << 10;
                    int align_bo_size = (((int)instructionsVec_[i].size() + (alignment - 1)) / alignment) * alignment;

                    pdi_instructions_.push_back(xrt::bo(instructions_, align_bo_size, offset));
                    LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                        << " DPU runner pdi instruction, bo address: " << pdi_instructions_[i].address()
                        << ",  bo size: " << align_bo_size;
                    pdi_instructions_[i].write(instructionsVec_[i].data(), instructionsVec_[i].size(), 0);
                    pdi_instructions_[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    offset += align_bo_size;
                }
                if (attrs->has_attr("bo_sram"))
                    attrs->set_attr<size_t>("bo_offset", offset);
            }
            else {
                pdi_instructions_.push_back(instructions_);
                pdi_instructions_[0].write(instructionsVec_[0].data(), instructionsVec_[0].size(), 0);
                pdi_instructions_[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

            }
        }
        else {
            int pdi_iter = 0;
            auto child_order_pdi = subgraph->children_topological_sort();
            size_t offset = 0;
            if (attrs->has_attr("bo_sram"))
                if (attrs->has_attr("bo_offset")) offset = attrs->get_attr<size_t>("bo_offset");
            for (const auto& child_iter_pdi : child_order_pdi) {
                std::vector<std::string> child_order;
                std::set<const xir::Subgraph*> children;

                if (!en_pdi) {
                    if (subgraph->has_attr("children_topological_sort"))
                        child_order = subgraph->get_attr<std::vector<std::string>>("children_topological_sort");
                    children = subgraph->get_children();
                }
                else {
                    if (child_iter_pdi->has_attr("children_topological_sort"))
                        child_order = child_iter_pdi->get_attr<std::vector<std::string>>("children_topological_sort");
                    children = child_iter_pdi->get_children();
                }
                if (child_order.size()) {
                    for (const auto& child_name : child_order) {
                        auto child = std::find_if(children.begin(), children.end(),
                            [&child_name](auto g) { return g->get_name() == child_name; });
                        if (child == children.end()) {
                            throw std::runtime_error("Error: missing child layer info in xmodel, please enable debug mode in xmodel compile");
                        }
                        if ((*child)->has_attr("mc_code")) {
                            auto& mc_code = (*child)->get_attr<std::vector<char> >("mc_code");
                            if (mc_code.size() == 0)
                                continue;
                        }
                        super_layer_init(pdi_iter, subgraph, *child, hw_context, kernel, offset, input_layer_index, output_layer_index);
                        input_layer_index += (*child)->get_input_tensors().size();
                        output_layer_index += (*child)->get_output_tensors().size();
                    }
                }
                else {
                    auto super_childs = child_iter_pdi->children_topological_sort();
                    std::vector<const xir::Subgraph*> subgs;
                    if (!en_pdi)
                        subgs = child_order_pdi;
                    else
                        subgs = super_childs;
                    for (const auto& child_iter : subgs) {
                        if (child_iter->has_attr("mc_code")) {
                            auto& mc_code = child_iter->get_attr<std::vector<char> >("mc_code");
                            if (mc_code.size() == 0)
                                continue;
                        }
                        super_layer_init(pdi_iter, subgraph, child_iter, hw_context, kernel, offset, input_layer_index, output_layer_index);
                        input_layer_index += child_iter->get_input_tensors().size();
                        output_layer_index += child_iter->get_output_tensors().size();
                    }

                }
 
                pdi_iter++;
                if (!en_pdi)
                    break; // only need loop one time when pdi disable

            }
            if (attrs->has_attr("bo_sram"))
                attrs->set_attr<size_t>("bo_offset", offset);
        }
        if (addr_patch_needed)
        {
            uint64_t input_addr = 0;
            input_addr = input_.address();

            // If enable_pad_control_packet is not set, assume padding is used (PHX behavior).
            // If enable_pad_control_packet is set, assume padding is used only if it is set to true.
            const bool pad_control_packet = subgraph->has_attr("enable_pad_control_packet")
                ? subgraph->get_attr<bool>("enable_pad_control_packet")
                : true;
            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << " DPU runner shim DMA patch input_addr: " << input_addr
                << ",  outut_addr: " << output_.address()
                << ",  mc_size: " << boMcVec.size();

            patchMcCodeDDR(input_addr + DDR_AIE_ADDR_OFFSET,
                parameters_.address() + DDR_AIE_ADDR_OFFSET,
                output_.address() + DDR_AIE_ADDR_OFFSET,
                intermediate_.address() + DDR_AIE_ADDR_OFFSET,
                bo_mc_.map<uint32_t*>(),
                boMcVec.size(),
                pad_control_packet);
            bo_mc_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        else if (args_num_ > BO_MC_ARG) {
            bo_mc_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        inputsOwned_ = create_inputs();
        outputsOwned_ = create_outputs();
        inputs_ = to_raw(inputsOwned_);
        outputs_ = to_raw(outputsOwned_);

        bypassCpu_ = attrs_->has_attr("bypass_cpu") && attrs_->get_attr<bool>("bypass_cpu");

        if (!bypassCpu_) {
            initializeDepadOutputTensors();
        }
    }

    /**
     * Destroy DpuKernelRunner object.
     */
    virtual ~DpuKernelRunner() {
#ifdef _WIN32
        TL_TRACE("DpuKernelRunner::~DpuKernelRunner");
#endif
    }

    void initializeDepadOutputTensors()
    {
        for (auto& tensor : output_tensors_) {
            auto srcElements = tensor->get_element_num();
            auto srcShapesXir = tensor->get_shape();
            auto srcStridesXir = tensor->get_attr<std::vector<std::int32_t>>("stride");
            int element_stride = tensor->get_data_type().bit_width / 8;

            std::vector<int32_t> srcShapes;
            std::vector<int32_t> srcStrides;

            auto shapeSize = srcShapesXir.size();

            if (shapeSize < 4) {
                srcShapes = std::vector<int32_t>(4 - srcShapesXir.size(), 1);
                srcStrides = std::vector<int32_t>(4 - srcShapesXir.size(), srcStridesXir[0]);
                std::copy(srcShapesXir.begin(), srcShapesXir.end(), std::back_inserter(srcShapes));
                std::copy(srcStridesXir.begin(), srcStridesXir.end(), std::back_inserter(srcStrides));
            }
            else {
                srcShapes = srcShapesXir;
                srcStrides = srcStridesXir;
            }

            if (srcShapes.size() > 4) {
                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner output depad, tensor name: " << tensor->get_name()
                    << ",  shape dimension = " << srcShapes.size();
                throw std::runtime_error("Now depad only supoort output dimension <= 4, we got " + tensor->get_name() + " dimension = " + std::to_string(srcShapes.size()));
            }

            srcShapes[3] *= element_stride;
            srcStrides[0] *= element_stride;
            srcStrides[1] *= element_stride;
            srcStrides[2] *= element_stride;
            bool isDepaddingNeeded = true;
            if ((srcShapes[0] == 1) && (srcStrides.size() == 4) && (srcShapes.size() == 4)) {
                if ((srcStrides[3] == 1) && (srcStrides[2] == srcShapes[3]) && (srcStrides[1] == (srcShapes[2] * srcShapes[3])))
                    isDepaddingNeeded = false;
            }

            LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                << " DPU runner output depad, tensor name: "<< tensor->get_name()
                << ",  depad needed: "<< isDepaddingNeeded
                << ",  dims: " << srcStrides.size()
                << ",  num: "<< srcElements
                << ",  stride[0]: " << srcStrides[0]
                << ",  ddr_addr: " << tensor->get_attr<std::int32_t>("ddr_addr");

            if (isDepaddingNeeded) {
                if (tensor->get_attr<int>("reg_id") == 2) {
                    depad_output_tensors_.emplace_back(input_,srcShapes, srcStrides, tensor->get_attr<std::int32_t>("ddr_addr"));
                }
                else if (tensor->get_attr<int>("reg_id") == 3) {
                    depad_output_tensors_.emplace_back(output_, srcShapes, srcStrides, tensor->get_attr<std::int32_t>("ddr_addr"));
                }
                else {
                    throw std::runtime_error("Invalid reg id for this tensor: " + tensor->get_attr<int>("reg_id"));
                }
            }
        }
    }

    void depad(xir::Tensor* tensor, char* src, char* dst) {
        uint64_t buff_num;
        uint64_t elem_num = tensor->get_element_num();
        std::pair<int, int>  c_cnt;
        auto shapes = tensor->get_shape();
        int element_stride = tensor->get_data_type().bit_width / 8;
        if (tensor->has_attr("stride")) {
            auto strides = tensor->get_attr<std::vector<int>>("stride");
            buff_num = (uint64_t)strides[0] * (uint64_t)shapes[0];
            if (buff_num > elem_num) {
                int bCnt = 1;
                for (unsigned c = 0; c < shapes.size(); c++) {
                    bCnt *= shapes[shapes.size() - 1 - c];
                    if (bCnt != strides[c]) {
                        c_cnt = std::make_pair(c, bCnt);
                        break;
                    }
                    if ((c+1) == shapes.size())
                        c_cnt = std::make_pair(shapes.size() - 1, elem_num);
                }
            }
        }
        else {
            buff_num = elem_num;
        }
        if (buff_num > elem_num) {
            auto strides = tensor->get_attr<std::vector<int>>("stride");
            int32_t srcIdx;
            int32_t dstIdx = 0;
            //int32_t dimsSize = shapes.size();
            std::vector<int32_t> srcIndexes(shapes.size(), 0);
            while (1) {
                srcIdx = 0;
                for (unsigned c = 0; c < shapes.size(); c++) {
                    srcIdx += srcIndexes[c] * strides[c];
                }
                memcpy(dst + dstIdx * element_stride, src + srcIdx * element_stride, c_cnt.second * element_stride); // output with dirty, so need to remove.
                dstIdx +=c_cnt.second;
                int j;
                for (j = shapes.size() - c_cnt.first - 2; j >= 0; j--) {
                    srcIndexes[j]++;
                    if (srcIndexes[j] < shapes[j])
                        break;
                    srcIndexes[j] = 0;
                }
                if (j < 0)
                    break;
            }
        }
    }
    void dump_io(address_info_t input, bool isOutput) {
        xrt::bo fbo;
        auto reg_id = std::get<3>(input);
        auto layer_index = std::get<4>(input);
        std::string fusion_memory_status = std::get<5>(input);
        if (reg_id == 2)
            fbo = input_;
        else if (reg_id == 3)
            fbo = output_;
        else if (reg_id == 1)
            fbo = intermediate_;
        auto offset = std::get<0>(input);
        auto tensor = std::get<1>(input);
        auto stride = tensor->get_attr<std::vector<std::int32_t>>("stride");
        int element_stride = tensor->get_data_type().bit_width / 8;
        bool need_depad = false;
        auto shape = tensor->get_shape();
        if (stride[0] > tensor->get_element_num()) {//only stride[0] > real_tensor size, need check depad_need
            need_depad = true;

            int idx = 0;
            for (unsigned i = 0; i < shape.size(); i++) { // find the first dimension not equal to 1
                if (shape[i] != 1) {
                    idx = i;
                    break;
                }
            }
            if (idx != (stride.size() - 1)) { //if [1,1,x,1] [1,x,1,1] [1,x,1] no need depad
                int32_t tmp_size = 1;
                for (unsigned j = idx + 1; j < shape.size(); j++)
                    tmp_size *= shape[j];
                if (tmp_size == stride[idx])
                    need_depad = false;
            }
            else { //last dimenstion no need depad e.g  [1, x] [1,1,x] [1,1,1,x]
                need_depad = false;
            }
        }
        int size = tensor->get_data_size();
        if (need_depad)
            size = stride[0] * element_stride;
        auto data = std::make_unique<char[]>(size);
        auto name =  std::get<2>(input);
        size_t pos = 0;
        for (auto i = 0U; i < name.size(); i++) {
            if (name[i] == '/') name[i] = '_';
        }
        std::vector<std::pair<std::string::size_type, std::string::size_type>> bracket_indices;
        for (auto i = 0U; i < name.size(); i++) {
            if (name[i] == '(' && (bracket_indices.empty() || bracket_indices.back().second != 0U)) {
            std::pair<std::string::size_type, std::string::size_type> bracket_pair(i, 0U);
            bracket_indices.push_back(bracket_pair);
            } else if (name[i] == ')' && !bracket_indices.empty()) {
            bracket_indices.back().second = i;
            } 
        }
        std::string prefix = "subgraph_";
        for (std::string::size_type i = bracket_indices.size(); i > 0U; i--) {
            if (bracket_indices[i - 1].first == 0U || 
                (name.rfind(prefix, 0) == 0 && bracket_indices[i - 1].first == prefix.size())) {
            // protected tensor name: erase the brackets
            name.erase(bracket_indices[i - 1].second, 1);
            name.erase(bracket_indices[i - 1].first, 1);
            } else {
            // erase the contents
            name.erase(bracket_indices[i - 1].first, bracket_indices[i - 1].second - bracket_indices[i - 1].first + 1);
            }
        }        
        auto dump_with_superlayer_id = (bool)stoi(get_env_var("XLNX_DUMP_WITH_SUPERLAYER_ID", "0"));
        if(dump_with_superlayer_id)
            name = std::string("superlayer-") + std::to_string(layer_index) + std::string("_") + name;
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << " DPU runner dump tensor name: " << name
            << ", fusion_memory_status: " << fusion_memory_status
            << ", super_layer_index: " << layer_index
            << ",  reg_id: " << reg_id;
        if(fusion_memory_status == "ON_MT")
            return;
        if (reg_id == 0xff) //TODO this is for short skip, in the future, check tensor reg
            return;
        std::stringstream ss;

        fbo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);
        if (tensor->has_attr("ddr_layout") && tensor->get_attr<std::int32_t>("ddr_layout") && (shape.size() == 4)) {
            // N H C\8 W 8  -->  N H W C
            auto data_tmp = std::make_unique<char[]>(size);
            memcpy(data_tmp.get(), ((char*)fbo.map<void*>()) + offset, size);
            for (int n = 0; n < shape[0]; n++) { // n = 1
                for (int h = 0; h < stride[0]/stride[1]; h++) {
                    for (int w = 0; w < stride[1]/stride[2]; w++) {
                        for (int c = 0; c < stride[2]; c += 8) {
                            int src_idx = n * stride[0] + h * stride[1] + c * stride[1]/stride[2] + w * 8;
                            int dst_idx = n * stride[0] + h * stride[1] + w * stride[2] + c;
                            memcpy(data.get() + dst_idx, data_tmp.get() + src_idx, 8);
                        }
                    }
                }
            }
        }
        else {
            memcpy(data.get(), ((char*)fbo.map<void*>()) + offset, size);
        }
        if (need_depad)
            depad(tensor, data.get(), data.get());

        std::ofstream(dump_folder_ / name, std::ios::binary).write(data.get(), tensor->get_data_size());
        //std::ofstream(dump_folder_ / std::get<2>(input)).close();
    }
    void run_pdi(int pdi_iter) {
        if (!debug_mode_) {  //normal node run

            trigger_run(run_vec_[pdi_iter], pdi_instructions_[pdi_iter], instructionWords_[pdi_iter], dbg_layers_[pdi_iter][0].name);
        }
        else { // super layer run
            // liter super layers
            for (auto iter = dbg_layers_[pdi_iter].begin() + 1; iter != dbg_layers_[pdi_iter].end(); iter++) {
                auto layer = *iter;
                auto code_bo = std::get<0>(layer.code_addr);
                vitis::ai::trace::add_trace("dpu-runner", layer.name, 1, layer.workload, layer.depth);
                vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_start, 0, 0);
                if (code_bo.size() > 0) {
                    trigger_run(run_vec_[pdi_iter], code_bo, code_bo.size() / sizeof(int), layer.name);
                }
                vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_end, 0, 0);
                if (dump_mode_) {
                    for (auto& out : layer.outputs) {
                        dump_io(out, true);
                    }
                }
            }
        }
    }
    void trigger_run(xrt::run &xrt_run, xrt::bo &code, uint32_t size, const std::string& name) {
        if (ENV_PARAM(XLNX_XRT_CU_DRY_RUN)) {
            return;
        }

        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << " DPU runner kernel run, input: " << input_.address()
            << ",  out: " << output_.address()
            << ",  parameter: " << parameters_.address()
            << ",  inter: " << intermediate_.address()
            << ",  code: " << code.address();
        __TIC__(KERNEL_RUN)
        if (args_num_ > BO_MC_ARG) {
            xrt_run(
                std::uint64_t(1), // Indicate to FW not to run selfTest
                input_,
                parameters_,
                output_,
                intermediate_,
                code,
                size,
                bo_mc_
            );
        }
        else {
            xrt_run(
                std::uint64_t(1), // Indicate to FW not to run selfTest
                input_,
                parameters_,
                output_,
                intermediate_,
                code,
                size
            );
        }
        try {
            xrt_run.wait2();
        }
        catch (const std::exception& e) {
            LOG(FATAL) << "DPU timeout: "
                << "(Exception type: " << typeid(e).name() << "), Timeout layer name:[" << name << "], " << e.what();
        }
        __TOC__(KERNEL_RUN)
        LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
            << " DPU runner kernel done";
    }

    /**
     * Run inference for this DpuKernelRunner.
     *
     * @param inputs
     *  Vector of tensor buffer pointers which provide the buffers to process. Data source.
     * @param outputs
     *  Vector of tensor buffer pointers which provide the buffers where data must be output. Data sink.
     *
     * Concrete KernelRunner implementations must define this function.
     * DpuKernelRunner will run inference on the DPU.
     */
    virtual void run(const std::vector<vart::TensorBuffer*>& inputs, const std::vector<vart::TensorBuffer*>& outputs) override
    {
        if(padNeeded_) {
            __TIC__(INPUT_PAD)
            for (auto &input : inputs) {
                // Get the raw pointers, and the size in bytes of io buffers
                std::uint64_t inputRaw;
                size_t inputSize;
                std::tie(inputRaw, inputSize) = input->data();

                const auto &tensorName = input->get_tensor()->get_name();
                auto &tensorInfo = input_tensor_info_map_[tensorName];

                LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
                    << " DPU runner input pad for " << tensorName
                    << ", pad needed " << tensorInfo.padNeeded
                    << ", input tensor reg = " << input->get_tensor()->get_attr<int>("reg_id");

                std::uint8_t *outputPtr = tensorInfo.paddedSubBo.map<std::uint8_t *>();
                if (tensorInfo.padNeeded) {
                    fastPad(reinterpret_cast<const std::uint8_t *>(inputRaw), outputPtr, tensorInfo.paddedOutputShape, tensorInfo.dstSlices);
                }
            }
            __TOC__(INPUT_PAD)
        }
        __TIC__(SYNC_INPUT_BO)
        input_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        output_.sync(XCL_BO_SYNC_BO_FROM_DEVICE); // Flush and Invalidate Cached BO
        __TOC__(SYNC_INPUT_BO)

        if (!debug_mode_) {
            vitis::ai::trace::add_trace("dpu-runner", dbg_layers_[0][0].name, 1, dbg_layers_[0][0].workload, dbg_layers_[0][0].depth);
            vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_start, 0, 0);
        }
        if (dump_mode_) {
            auto& inputs_l = dbg_layers_[0][0].inputs;
            for (auto& input : inputs_l) {
                dump_io(input, false);
            }
        }
        for (int pdi_iter = 0; pdi_iter < run_vec_.size(); pdi_iter++) {
            // Execute Kernel
            run_pdi(pdi_iter);
        } // pdi_iter
        if (!debug_mode_) {
            if (dump_mode_) {
                for (auto& out : dbg_layers_[0][0].outputs) {
                    dump_io(out, true);
                }
            }
            vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_end, 0, 0);
        }
        __TIC__(SYNC_OUTPUT_BO)
#ifdef _WIN32
        output_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
#endif
        __TOC__(SYNC_OUTPUT_BO)
        // Probably should wrap this in a function call
        // However, the point here is to convert uncontiguous output into a contiguous/dense tensor
        // Don't do this, if application wants to bypass it
        __TIC__(OUTPUT_DEPAD)
        if (!bypassCpu_)
        {
            for (auto &tensor : depad_output_tensors_) {
                fastDepad(tensor.shapes, tensor.strides, tensor.output_bo.map<std::uint8_t*>() + tensor.ddr_offset);
            }
        }
        __TOC__(OUTPUT_DEPAD)
    }

    /**
     * Construct and return TensorBuffers for this DpuKernelRunner.
     *
     * @return
     *  Vector of unique_ptr of TensorBuffers.
     */
    virtual std::vector<std::unique_ptr<vart::TensorBuffer>> create_inputs() override
    {
        std::vector<std::unique_ptr<vart::TensorBuffer>> inputs;
        std::transform(input_tensors_.cbegin(), input_tensors_.cend(), std::back_inserter(inputs),
            [this](const xir::Tensor* tensor)
            {
                auto ddr_addr = tensor->get_attr<std::int32_t>("ddr_addr");
                auto& strides = tensor->get_attr<std::vector<std::int32_t>>("stride");
                if ((!padNeeded_) || (!input_tensor_info_map_[tensor->get_name()].padNeeded)) {
                if (tensor->get_attr<int>("reg_id") == 2) {

                    return std::make_unique<XrtBuffer>(tensor, strides, input_, tensor->get_data_size(), ddr_addr);

                }
                else if (tensor->get_attr<int>("reg_id") == 3) {
                    return std::make_unique<XrtBuffer>(tensor, strides, output_, tensor->get_data_size(), ddr_addr);
                }
                else {
                    throw std::runtime_error("Invalid reg id for this tensor: " + tensor->get_attr<int>("reg_id"));
                }
            }
            else {
                return std::make_unique<XrtBuffer>(tensor, strides, pad_input_, tensor->get_data_size(), input_tensor_info_map_[tensor->get_name()].padOffset);
            }
        });

        return inputs;
    }

    /**
     * Construct and return TensorBuffers for this DpuKernelRunner.
     *
     * @return
     *  Vector of unique_ptr of TensorBuffers.
     */
    virtual std::vector<std::unique_ptr<vart::TensorBuffer>> create_outputs() override
    {
        std::vector<std::unique_ptr<vart::TensorBuffer>> outputs;
        std::transform(output_tensors_.cbegin(), output_tensors_.cend(), std::back_inserter(outputs),
            [this](const xir::Tensor* tensor)
        {
            auto ddr_addr = tensor->get_attr<std::int32_t>("ddr_addr");
            auto& strides = tensor->get_attr<std::vector<std::int32_t>>("stride");
            return std::make_unique<XrtBuffer>(tensor, strides, output_, tensor->get_data_size(), ddr_addr);
        });

        return outputs;
    }

    /**
     * Get this DpuKernelRunner's kernel name.
     *
     * @return
     *  This DpuKernelRunner's kernel name as std::string.
     *
     * Note that currently the name of kernel in xclbin
     * does not match the kernel name in the xmodel, so this function is overriden
     */
    virtual std::string get_kernel_name() override
    {
        return { "DPU_1x4" };
    }
    using address_info = std::tuple<uint64_t, uint64_t, std::string, int32_t>;

    using address_info_code = std::tuple<xrt::bo, int32_t, std::string>;
    struct layer_info {
        layer_info(std::string name) {
            this->name = name;
            this->depth = 0;
            this->workload = 0;
        }
        std::string name;
        uint64_t workload;
        uint64_t depth;
        void set_depth(uint64_t value) { this->depth = value; }
        void set_workload(uint64_t value) { this->workload = value; }
        // address_info_code is for save super layer mc_code bo
        address_info_code code_addr;
        std::vector<address_info_t> inputs;
        std::vector<address_info_t> outputs;

        static std::string name_map(std::string raw) {
            size_t pos;
            std::string tmp = raw;
            while ((pos = tmp.find("/")) != std::string::npos) {
                tmp.replace(pos, 1, "_");
            }
            while ((pos = tmp.find(":")) != std::string::npos) {
                tmp.replace(pos, 1, "_");
            }
            return tmp + ".bin";
        }
    };
  virtual std::vector<vart::TensorBuffer *> get_inputs() override { return inputs_; }

  virtual std::vector<vart::TensorBuffer *> get_outputs() override { return outputs_; }
  /**
   * use xir::Attrs to pass info to RunnerExt at runtime.
   * return 0: good; 1: error, RunnerExt cannot find any useful info in Attrs
   */
  virtual int set_run_attrs(std::unique_ptr<xir::Attrs>&) override { return 0; }

protected:
    xir::Attrs* attrs_;
    xrt::bo instructions_;
    xrt::bo parameters_;
    xrt::bo bo_mc_;
    xrt::bo intermediate_;
    xrt::bo input_;
    xrt::bo output_;
    std::vector<xrt::bo> pdi_instructions_;
    std::vector<std::vector<char> > instructionsVec_;
    xrt::kernel pdi_kernel;
    std::vector<xrt::run> run_vec_;

    std::vector<std::uint32_t> instructionWords_;

    std::int32_t inputSize_;
    std::int32_t outputSize_;
    std::int32_t intermediateSize_;

    //if padding needed
    bool padNeeded_;
    xrt::bo pad_input_;

    std::unordered_map<std::string, TensorInfo> input_tensor_info_map_;

    int args_num_{ 0 };

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputsOwned_;
    std::vector<std::unique_ptr<vart::TensorBuffer>> outputsOwned_;
    std::vector<vart::TensorBuffer*> inputs_;
    std::vector<vart::TensorBuffer*> outputs_;
    std::vector<std::vector<layer_info>> dbg_layers_;
    bool debug_mode_;
    bool dump_mode_;
    std::string sg_name_;
    fs::path dump_folder_;
    std::int32_t padBufferSize_;

    bool bypassCpu_;

    std::vector<DepadTensorInfo> depad_output_tensors_;
};
