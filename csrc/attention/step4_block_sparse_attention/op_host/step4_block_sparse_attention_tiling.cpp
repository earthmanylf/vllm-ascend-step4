/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "step4_block_sparse_attention_tiling.h"

#include <cmath>

#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;

namespace optiling {
namespace {

constexpr uint32_t TND_DIM_NUM = 3;
constexpr size_t STEP4_BSA_DEFAULT_WORKSPACE = 16 * 1024 * 1024;
// Must match STEP4_SEG_LIST_WORDS in packed_region_gather_arch35.hpp.
constexpr uint32_t STEP4_SEG_LIST_WORDS = 208;
constexpr uint32_t STEP4_SEG_LIST_BYTES = STEP4_SEG_LIST_WORDS * static_cast<uint32_t>(sizeof(uint32_t));
constexpr uint32_t TILE_128 = 128;
constexpr uint32_t TILE_256 = 256;
constexpr uint32_t Q_BASE_TILE = 16;
constexpr uint32_t KV_BASE_TILE = 256;
constexpr uint32_t SOLO_BUF = 1;
constexpr uint32_t DUO_BUF = 2;
constexpr uint32_t TRIO_BUF = 3;

inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (align == 0) ? value : ((value + align - 1) / align * align);
}

} // namespace

ge::graphStatus Step4Tiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t usedAic = (aicNum == 0) ? 1U : aicNum;
    const uint32_t usedAiv = (aivNum == 0) ? usedAic : aivNum;
    aicCoreNum_ = usedAic;
    blockDim_ = ascendcPlatform.CalcTschBlockDim(usedAiv, usedAic, usedAiv);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Step4Tiling::ParseInputTensors(gert::TilingContext *context)
{
    const gert::StorageShape *queryShape = context->GetInputShape(STEP4_BSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::StorageShape *keyShape = context->GetInputShape(STEP4_BSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::StorageShape *packedShape = context->GetInputShape(STEP4_BSA_IDX_PACKED_REGIONS);
    OP_CHECK_NULL_WITH_CONTEXT(context, packedShape);

    // packed_regions dtype selects the encoding contract: INT32 = Prefill, INT64 = Decode.
    const auto *packedDesc = context->GetInputDesc(STEP4_BSA_IDX_PACKED_REGIONS);
    OP_CHECK_NULL_WITH_CONTEXT(context, packedDesc);
    const ge::DataType packedDtype = packedDesc->GetDataType();
    OP_CHECK_IF(packedDtype != ge::DT_INT32 && packedDtype != ge::DT_INT64,
                OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention",
                                            "packed_regions dtype must be INT32 (Prefill) or INT64 (Decode)."),
                return ge::GRAPH_FAILED);
    isDecode_ = (packedDtype == ge::DT_INT64);

    // QKV dtype: BF16 or FP16, and query/key/value must share the same dtype.
    const auto *queryDesc = context->GetInputDesc(STEP4_BSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    const auto *keyDesc = context->GetInputDesc(STEP4_BSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    const auto *valueDesc = context->GetInputDesc(STEP4_BSA_IDX_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueDesc);
    const ge::DataType queryDtype = queryDesc->GetDataType();
    const ge::DataType keyDtype = keyDesc->GetDataType();
    const ge::DataType valueDtype = valueDesc->GetDataType();
    OP_CHECK_IF(queryDtype != ge::DT_BF16 && queryDtype != ge::DT_FLOAT16,
                OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention",
                                            "query dtype must be BF16 or FP16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryDtype != keyDtype || queryDtype != valueDtype,
                OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention",
                                            "query/key/value must share the same dtype."),
                return ge::GRAPH_FAILED);
    isFp16_ = (queryDtype == ge::DT_FLOAT16);

    const gert::Shape &q = queryShape->GetStorageShape();
    const gert::Shape &k = keyShape->GetStorageShape();
    qTokens_ = (q.GetDimNum() >= 1) ? static_cast<uint32_t>(q.GetDim(0)) : 0;
    kvTokens_ = (k.GetDimNum() >= 1) ? static_cast<uint32_t>(k.GetDim(0)) : 0;
    embeddingSize_ = (q.GetDimNum() >= TND_DIM_NUM) ? static_cast<uint32_t>(q.GetDim(2)) : 0;
    const gert::Shape &packed = packedShape->GetStorageShape();
    topK_ = (packed.GetDimNum() >= 2) ? static_cast<uint32_t>(packed.GetDim(1)) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Step4Tiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *numHeadsPtr = attrs->GetInt(STEP4_BSA_ATTR_NUM_HEADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsPtr);
    const int64_t *kvHeadsPtr = attrs->GetInt(STEP4_BSA_ATTR_NUM_KV_HEADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvHeadsPtr);
    const int64_t *regionSizePtr = attrs->GetInt(STEP4_BSA_ATTR_REGION_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, regionSizePtr);
    const float *scalePtr = attrs->GetFloat(STEP4_BSA_ATTR_SCALE_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalePtr);

    numHeads_ = static_cast<uint32_t>(*numHeadsPtr);
    kvHeads_ = static_cast<uint32_t>(*kvHeadsPtr);
    groupSize_ = (kvHeads_ == 0) ? 0 : (numHeads_ / kvHeads_);
    OP_CHECK_IF(*regionSizePtr <= 0 || *regionSizePtr > 255,
                OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention",
                                            "region_size must be in [1, 255] (Prefill valid_tokens is 8 bits), got %ld.",
                                            *regionSizePtr),
                return ge::GRAPH_FAILED);
    regionSize_ = static_cast<uint32_t>(*regionSizePtr);
    const int64_t *pageSizePtr = attrs->GetInt(STEP4_BSA_ATTR_PAGE_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, pageSizePtr);
    int64_t pageSizeRaw = *pageSizePtr;
    if (pageSizeRaw <= 0) {
        pageSizeRaw = static_cast<int64_t>(STEP4_BSA_DEFAULT_PAGE_SIZE);
    }
    OP_CHECK_IF(pageSizeRaw % *regionSizePtr != 0,
                OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention",
                                            "page_size must be a multiple of region_size, got page_size=%ld "
                                            "region_size=%ld.",
                                            pageSizeRaw, *regionSizePtr),
                return ge::GRAPH_FAILED);
    pageSize_ = static_cast<uint32_t>(pageSizeRaw);
    scaleValue_ = *scalePtr;
    if (scaleValue_ == 0.0f && embeddingSize_ > 0) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Step4Tiling::CheckKvSeqlens(gert::TilingContext *context)
{
    if (!isDecode_) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *seqShape = context->GetOptionalInputShape(STEP4_BSA_IDX_KV_SEQLENS);
    OP_CHECK_IF(seqShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention", "Decode requires kv_seqlens."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Step4Tiling::CalculateWorkSpace(gert::TilingContext *context)
{
    context->SetBlockDim(blockDim_);
    uint64_t tokens = static_cast<uint64_t>(topK_) * static_cast<uint64_t>(regionSize_);
    uint32_t maxTiles = 1;
    if (KV_BASE_TILE > 0) {
        maxTiles = static_cast<uint32_t>((tokens + KV_BASE_TILE - 1) / KV_BASE_TILE);
    }
    if (maxTiles == 0) {
        maxTiles = 1;
    }
    maxKvTiles_ = maxTiles;
    uint64_t nTask = static_cast<uint64_t>(qTokens_) * static_cast<uint64_t>(kvHeads_);
    if (nTask == 0) {
        nTask = 1;
    }
    uint64_t listBytes = nTask * static_cast<uint64_t>(maxTiles) * STEP4_SEG_LIST_BYTES;
    uint64_t userBytes = STEP4_BSA_DEFAULT_WORKSPACE;
    if (listBytes > userBytes) {
        userBytes = listBytes;
    }
    workSpaceSize_ = libApiWorkSpaceSize_ + userBytes;
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(workSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Step4Tiling::FillTilingData(gert::TilingContext *context)
{
    qBaseTile_ = AlignUp((groupSize_ == 0) ? Q_BASE_TILE : groupSize_, Q_BASE_TILE);
    qAligned128_ = AlignUp(qBaseTile_, TILE_128);
    dAligned128_ = AlignUp(embeddingSize_, TILE_128);
    kvAligned128_ = AlignUp(KV_BASE_TILE, TILE_128);
    totalTaskNum_ = qTokens_ * kvHeads_;

    tilingData_->set_qTokens(qTokens_);
    tilingData_->set_kvTokens(kvTokens_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_regionSize(regionSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_qBaseTile(qBaseTile_);
    tilingData_->set_kvBaseTile(KV_BASE_TILE);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_mm1L1TileM(qAligned128_);
    tilingData_->set_mm1L1TileN(TILE_128);
    tilingData_->set_mm1L1TileKLeft(dAligned128_);
    tilingData_->set_mm1L1TileKRight(TILE_256);
    tilingData_->set_mm2L1TileM(qAligned128_);
    tilingData_->set_mm2L1TileN(TILE_256);
    tilingData_->set_mm2L1TileKLeft(kvAligned128_);
    tilingData_->set_mm2L1TileKRight(kvAligned128_);
    tilingData_->set_qL1BufNum(SOLO_BUF);
    tilingData_->set_kL1BufNum(DUO_BUF);
    tilingData_->set_vL1BufNum(SOLO_BUF);
    tilingData_->set_pL1BufNum(TRIO_BUF);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_pageSize(pageSize_);
    tilingData_->set_maxKvTiles(maxKvTiles_);

    context->SetTilingKey(GenerateTilingKey());
    return ge::GRAPH_SUCCESS;
}

uint64_t Step4Tiling::GenerateTilingKey() const
{
    if (isFp16_) {
        return isDecode_ ? STEP4_BSA_DECODE_FP16_TILING_KEY : STEP4_BSA_PREFILL_FP16_TILING_KEY;
    }
    return isDecode_ ? STEP4_BSA_DECODE_BF16_TILING_KEY : STEP4_BSA_PREFILL_BF16_TILING_KEY;
}

ge::graphStatus Step4Tiling::GetTiling(gert::TilingContext *context, Step4BlockSparseAttentionTilingData &tilingData)
{
    tilingData_ = &tilingData;
    if (GetNpuInfo(context) != ge::GRAPH_SUCCESS || ParseInputTensors(context) != ge::GRAPH_SUCCESS ||
        ParseAttrs(context) != ge::GRAPH_SUCCESS || CheckKvSeqlens(context) != ge::GRAPH_SUCCESS ||
        CalculateWorkSpace(context) != ge::GRAPH_SUCCESS || FillTilingData(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Step4Tiling::SetTilingData(gert::TilingContext *context,
                                           Step4BlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention", "RawTilingData got from GE context is null."),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

namespace {

ge::graphStatus TilingForStep4BlockSparseAttention(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Step4BlockSparseAttention", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    Step4BlockSparseAttentionTilingData tilingData;
    Step4Tiling tiling;
    if (tiling.GetTiling(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tiling.SetTilingData(context, tilingData);
}

static ge::graphStatus TilingPrepareForStep4BlockSparseAttention(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(Step4BlockSparseAttention)
    .Tiling(TilingForStep4BlockSparseAttention)
    .TilingParse<Step4BlockSparseAttentionCompileInfo>(TilingPrepareForStep4BlockSparseAttention);

} // namespace optiling
