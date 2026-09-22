/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef STEP4_BLOCK_SPARSE_ATTENTION_TILING_H
#define STEP4_BLOCK_SPARSE_ATTENTION_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr uint32_t STEP4_BSA_IDX_QUERY = 0;
constexpr uint32_t STEP4_BSA_IDX_KEY = 1;
constexpr uint32_t STEP4_BSA_IDX_VALUE = 2;
constexpr uint32_t STEP4_BSA_IDX_PACKED_REGIONS = 3;
constexpr uint32_t STEP4_BSA_IDX_REGION_COUNTS = 4;
constexpr uint32_t STEP4_BSA_IDX_KV_SEQLENS = 5;

constexpr uint32_t STEP4_BSA_ATTR_NUM_HEADS = 0;
constexpr uint32_t STEP4_BSA_ATTR_NUM_KV_HEADS = 1;
constexpr uint32_t STEP4_BSA_ATTR_REGION_SIZE = 2;
constexpr uint32_t STEP4_BSA_ATTR_PAGE_SIZE = 3;
constexpr uint32_t STEP4_BSA_ATTR_SCALE_VALUE = 4;
constexpr uint32_t STEP4_BSA_ATTR_LAYOUT_Q = 5;
constexpr uint32_t STEP4_BSA_ATTR_LAYOUT_KV = 6;
constexpr uint32_t STEP4_BSA_ATTR_NUM_SPLITS = 7;
constexpr uint32_t STEP4_BSA_ATTR_RETURN_LSE = 8;

constexpr uint64_t STEP4_BSA_PREFILL_BF16_TILING_KEY = 0;
constexpr uint64_t STEP4_BSA_DECODE_BF16_TILING_KEY = 1;
constexpr uint64_t STEP4_BSA_PREFILL_FP16_TILING_KEY = 2;
constexpr uint64_t STEP4_BSA_DECODE_FP16_TILING_KEY = 3;
constexpr uint32_t STEP4_BSA_DEFAULT_PAGE_SIZE = 128;

BEGIN_TILING_DATA_DEF(Step4BlockSparseAttentionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, qTokens);
TILING_DATA_FIELD_DEF(uint32_t, kvTokens);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, regionSize);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, qBaseTile);
TILING_DATA_FIELD_DEF(uint32_t, kvBaseTile);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, qL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, kL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, vL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, pL1BufNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, pageSize);
TILING_DATA_FIELD_DEF(uint32_t, maxKvTiles);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Step4BlockSparseAttention, Step4BlockSparseAttentionTilingData)

struct Step4BlockSparseAttentionCompileInfo {};

class Step4Tiling {
public:
    Step4Tiling() = default;
    ~Step4Tiling() = default;

    ge::graphStatus GetTiling(gert::TilingContext *context, Step4BlockSparseAttentionTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context, Step4BlockSparseAttentionTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ParseInputTensors(gert::TilingContext *context);
    ge::graphStatus ParseAttrs(gert::TilingContext *context);
    ge::graphStatus CheckKvSeqlens(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    ge::graphStatus FillTilingData(gert::TilingContext *context);
    uint64_t GenerateTilingKey() const;

    // Parsed from input tensors.
    uint32_t qTokens_ = 0;
    uint32_t kvTokens_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t topK_ = 0;
    bool isDecode_ = false;
    bool isFp16_ = false;

    // Parsed from attrs.
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t regionSize_ = 0;
    uint32_t pageSize_ = 0;
    float scaleValue_ = 0.0f;

    // Derived tile info.
    uint32_t qBaseTile_ = 0;
    uint32_t qAligned128_ = 0;
    uint32_t dAligned128_ = 0;
    uint32_t kvAligned128_ = 0;
    uint32_t totalTaskNum_ = 0;

    // Platform info.
    uint32_t blockDim_ = 0;
    uint32_t aicCoreNum_ = 0;
    uint32_t maxKvTiles_ = 0;
    uint64_t libApiWorkSpaceSize_ = 0;
    uint64_t workSpaceSize_ = 0;

    Step4BlockSparseAttentionTilingData *tilingData_ = nullptr;
};

} // namespace optiling

#endif
