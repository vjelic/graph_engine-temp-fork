#pragma once

#include <xir/graph/graph.hpp>

#define BASE_ADDRESS_LENGTH 48
#define REG_ID_LENGTH 4

constexpr uint32_t DUMMY_MC_CODE_BUFFER_SIZE = 16; // use in case buffer doesn't exist, in bytes

constexpr uint64_t DDR_AIE_ADDR_OFFSET = ((uint64_t)(0x80000000));

uint32_t getRegID(uint64_t BDData, uint64_t mask)
{
  // in this case reg ID mask = 0xF
  return (uint32_t)(((BDData >> (BASE_ADDRESS_LENGTH - REG_ID_LENGTH)) & mask));
}
uint64_t getBaseAddress(uint64_t BDData, uint64_t mask)
{
  // in this case base address mask = 0xFFFFFFFFFFF;
  return BDData & mask;
}
// ******************************************************************************************** //
  std::string getAddrPatchReg(const xir::Subgraph* subgraph)
  {
    if(subgraph->has_attr("reg_id_to_context_type_v2"))
    {
       auto regIdMap = subgraph->get_attr<std::map<std::string, std::string>>("reg_id_to_context_type_v2");
       for(auto &regId: regIdMap)
       {
         if(regId.second=="CODE")
         {
           return regId.first;
         }
       }
    }
    else
    {
      if(subgraph->has_attr("reg_id_to_context_type"))
      {
        auto regIdMap = subgraph->get_attr<std::map<std::string, std::string>>("reg_id_to_context_type");
       for(auto &regId: regIdMap)
       {
         if(regId.second=="CODE")
         {
           return regId.first;
         }
       }
       
      }
    }
    return "";
  }

#define IFM_TYPE 0x0
#define PARAM_TYPE 0x1
#define OFM_TYPE 0x2
#define INTER_TYPE 0x3

struct DDRDataStartAddr
{
  uint64_t ifmStartAddr;
  uint64_t paramStartAddr;
  uint64_t ofmStartAddr;
  uint64_t interStartAddr;
  DDRDataStartAddr()
      : ifmStartAddr(0),
        paramStartAddr(0),
        ofmStartAddr(0),
        interStartAddr(0) {}
};

#define DMA_BD_NUM 16
std::array<uint32_t, DMA_BD_NUM> DMABDx2RegAddr;

//  patch DDR address，this funtion is from interpreter in LX6.
int32_t patchDDRAddrFromLogicToPhysic(uint32_t &BDData1, uint32_t &BDData2,
                                      struct DDRDataStartAddr DDRAddr)
{
  uint32_t addrLow = BDData1;
  uint32_t addrHigh = (BDData2 & 0x00000FFF);
  uint32_t regID = ((BDData2 >> 12) & 0xf);
  uint64_t tensorAddr = ((((uint64_t)addrHigh) << 32) | addrLow);

  switch (regID)
  {
  case IFM_TYPE:
    tensorAddr += DDRAddr.ifmStartAddr;
    break;
  case PARAM_TYPE:
    tensorAddr += DDRAddr.paramStartAddr;
    break;
  case OFM_TYPE:
    tensorAddr += DDRAddr.ofmStartAddr;
    break;
  case INTER_TYPE:
    tensorAddr += DDRAddr.interStartAddr;
    break;
  default:
    break;
  }

  BDData1 = ((tensorAddr)&0xFFFFFFFFC); // unused 2-LSB
  BDData2 = ((BDData2 & 0xFFFF0000) | (tensorAddr >> 32));
  return 0;
}
uint32_t patchddrAddress(uint32_t *BDData, uint32_t len, uint32_t addr,
                         struct DDRDataStartAddr DDRAddr)
{
  // check if shim tile BD register contains DDR address.
  // This is to support variable number of DMA_BDx register configurations, but this function needs to be checked.
  // Now we write register from DMA_BDx_0 to DMA_BDx_7 every time, for more efficiency, we may only write part of eight DMA_BDx later.
  // One thing to note is that we cannot only write the Base_Address_High of DMA_BDx_2, which also means that the address of DMA_BDx_2
  // cannot be in the Local Byte Address of control packet(CP). So we start traversing from addr plus 4.
  // Taking DMA_BD0 as an examle, now we fully configure from 0x1D000 to 0x1D01C, later we may only config five registers,
  // say from 0x1D00C to 0x1D01C. the position of Base_Address_High in BD data is variable, and may even not exist.
  // so We need to check if the shim tile DMA_BDx register contains the DDR address.
  for (int i = 1; i < len + 1; i++)
  {
    addr += 4;
    if (DMABDx2RegAddr.end() != std::find(DMABDx2RegAddr.begin(), DMABDx2RegAddr.end(), addr))
    {
      // patch DDR Addrese from offset to phisical address
      patchDDRAddrFromLogicToPhysic(BDData[i - 1], BDData[i], DDRAddr);
    }
  }

  return 0;
}

int patchMcCodeDDR(uint64_t ddr_base_ifm, uint64_t ddr_base_param, uint64_t ddr_base_ofm,
                   uint64_t ddr_base_inter, uint32_t *mc_code_ddr, uint32_t mc_code_ddr_size_bytes,
                   int pad_control_packet)
{
  struct DDRDataStartAddr DDRAddr;
  DDRAddr.ifmStartAddr = ddr_base_ifm;
  DDRAddr.paramStartAddr = ddr_base_param;
  DDRAddr.ofmStartAddr = ddr_base_ofm;
  DDRAddr.interStartAddr = ddr_base_inter;

  // list all shim tile BD registers DDR address need to be processed
  for (int i = 0; i < 16; i++)
  {
    DMABDx2RegAddr[i] = 0x0001D008 + 0x20 * i;
  }

  uint32_t dataSize = 0;
  uint32_t localByteAddress = 0;
  uint32_t pc = 0;
  // Traverse all mc code ddr instructions
  while (pc < mc_code_ddr_size_bytes/4)
  {
    // read packet header and control packet, parse the data size and BD register addr
    pc += 2;  
    dataSize = ((mc_code_ddr[pc - 1] >> 20) & 0x3);
    localByteAddress = (mc_code_ddr[pc - 1] & 0xfffff);

    // patch shim tile register DMA_BDx DDR address
    patchddrAddress(&mc_code_ddr[pc], dataSize, localByteAddress, DDRAddr);
    pc += (dataSize + 1);

    // control packets aligned to 256 bit
    if (pad_control_packet)
    {
      pc += (8 - (pc%8))%8;
    }
  }

  return 0;
}
