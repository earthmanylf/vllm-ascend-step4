/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BSA_ARCH35_KERNEL_UTILS
#define BSA_ARCH35_KERNEL_UTILS

#include "../attn_infra/bsa_base_defs.hpp"
#include "../attn_infra/arch/bsa_arch.hpp"
#include "../attn_infra/layout/bsa_layout.hpp"

#include "../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../attn_infra/arch/bsa_resource.hpp"

#include "../tla/tensor_bsa.hpp"
#include "../tla/layout_bsa.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#include "../attn_infra/epilogue/block/bsa_block_epilogue.hpp"
#include "../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../attn_infra/gemm/block/bsa_block_mmad.hpp"
#include "../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../attn_infra/gemm/bsa_gemm_type.hpp"
#include "../attn_infra/gemm/block/packed_region_gather_arch35.hpp"

// 0: HEAD. Decode tile 0 before the FA loop; in-loop Vector produces tile t+1
//    while Cube Waits/QKs tile t (flag 15). Softmax still after QK. No DecodeAll.
#ifndef STEP4_ISSUE_DIAG
#define STEP4_ISSUE_DIAG 0
#endif

namespace BsaKernelArch35 {

struct Step4BsaTilingData {
    uint32_t qTokens;
    uint32_t kvTokens;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t regionSize;
    uint32_t topK;
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
    uint32_t totalTaskNum;
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;
    float scaleValue;
    uint32_t pageSize;
    uint32_t maxKvTiles;
};

struct Step4KernelParamsArch35 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR packedRegions;
    GM_ADDR regionCounts;
    GM_ADDR kvSeqlens;
    GM_ADDR o;
    GM_ADDR lse;
    GM_ADDR workSpace;
    GM_ADDR tiling;
};

// Task S under the packed contract: history slots are full regions; only the last live
// slot may be partial. One GM load on that last slot — gather still walks every slot for phys.
// Prefill: INT32 valid in the word (-1 skip → tail contributes 0). Decode: INT64
// start|(phys<<32), valid=clamp(kv_seqlen-start,0,rs).
template <bool IS_DECODE, class PackedT>
__aicore__ inline uint32_t CompactedPackedTokenCount(AscendC::GlobalTensor<PackedT> gPacked, uint32_t rowStart,
                                                     uint32_t count, uint32_t regionSize, int32_t kvSeqlen)
{
    if (count == 0) {
        return 0;
    }
    PackedT meta = gPacked.GetValue(rowStart + count - 1);
    uint32_t phys = 0;
    uint32_t tailValid = 0;
    NpuArch::Gemm::Block::ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, tailValid);
    (void)phys;
    if (tailValid > regionSize) {
        tailValid = regionSize;
    }
    return (count - 1) * regionSize + tailValid;
}

} // namespace BsaKernelArch35

#endif
