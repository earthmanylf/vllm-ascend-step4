/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file step4_block_sparse_attention.cpp
 * \brief C++ wrapper for aclnnStep4BlockSparseAttention
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DIM_T = 0;
constexpr int64_t DIM_N = 1;
constexpr int64_t DIM_D = 2;
constexpr int64_t QUERY_DIM_NUM = 3;
constexpr int64_t MAX_EMBEDDING_SIZE = 256;
constexpr int64_t DEFAULT_PAGE_SIZE = 128;
constexpr int64_t LSE_OUT = 1;

std::tuple<at::Tensor, at::Tensor> step4_block_sparse_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &packedRegions,
    const at::Tensor &regionCounts, int64_t numHeads, int64_t numKeyValueHeads,
    const c10::optional<at::Tensor> &kvSeqlens, int64_t regionSize, int64_t pageSize, double scaleValue,
    c10::string_view layoutQ, c10::string_view layoutKv, int64_t numSplits, bool returnSoftmaxLse)
{
    // 入参校验：dtype / shape / 模式约束
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
                "step4_block_sparse_attention only supports BF16/FP16 query, got ", query.scalar_type());
    TORCH_CHECK(key.scalar_type() == query.scalar_type(), "key should share query dtype, got ", key.scalar_type());
    TORCH_CHECK(value.scalar_type() == query.scalar_type(), "value should share query dtype, got ",
                value.scalar_type());
    TORCH_CHECK(query.dim() == QUERY_DIM_NUM, "query should be 3-D TND [T, N1, D], got dim ", query.dim());
    TORCH_CHECK(query.size(DIM_D) <= MAX_EMBEDDING_SIZE,
                "embedding size (head dim) should not exceed 256, got ", query.size(DIM_D));
    TORCH_CHECK(packedRegions.scalar_type() == at::kInt || packedRegions.scalar_type() == at::kLong,
                "packed_regions should be INT32 (Prefill) or INT64 (Decode), got ", packedRegions.scalar_type());
    TORCH_CHECK(regionCounts.scalar_type() == at::kInt, "region_counts should be INT32, got ",
                regionCounts.scalar_type());
    const bool isDecode = packedRegions.scalar_type() == at::kLong;
    if (kvSeqlens.has_value() && kvSeqlens->defined()) {
        TORCH_CHECK(kvSeqlens->scalar_type() == at::kInt, "kv_seqlens should be INT32, got ",
                    kvSeqlens->scalar_type());
    }
    if (isDecode) {
        TORCH_CHECK(kvSeqlens.has_value() && kvSeqlens->defined(),
                    "kv_seqlens is required in Decode mode (packed_regions is INT64).");
    }
    TORCH_CHECK(numHeads > 0, "num_heads should be greater than 0, got ", numHeads);
    TORCH_CHECK(numKeyValueHeads > 0, "num_key_value_heads should be greater than 0, got ", numKeyValueHeads);
    TORCH_CHECK(regionSize > 0 && regionSize <= 255,
                "region_size should be in [1, 255] (Prefill valid_tokens is 8 bits), got ", regionSize);
    TORCH_CHECK(numSplits == 1, "num_splits > 1 is not supported yet, got ", numSplits);

    at::Tensor attentionOut{nullptr};
    at::Tensor softmaxLse{nullptr};
    {
        // DeviceGuard：必须在申请输出之前，切到输入张量所在设备
        const auto localDevice = c10::Device(query.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        // Kernel skips empty packed rows and does not write O/LSE; zeros match the skip contract.
        attentionOut = at::zeros(query.sizes(), query.options());
        auto optsF32 = query.options().dtype(at::kFloat);
        if (returnSoftmaxLse) {
            // TND query [T, N1, D] → LSE [T, N1]
            softmaxLse = at::zeros({query.size(DIM_T), query.size(DIM_N)}, optsF32);
        } else {
            softmaxLse = at::empty({0}, optsF32);
        }
    }

    std::string layoutQStr(layoutQ);
    std::string layoutKvStr(layoutKv);
    const int64_t returnSoftmaxLseFlag = returnSoftmaxLse ? LSE_OUT : 0;
    if (pageSize <= 0) {
        pageSize = DEFAULT_PAGE_SIZE;
    }
    TORCH_CHECK(pageSize % regionSize == 0,
                "page_size must be a multiple of region_size, got page_size=", pageSize,
                " region_size=", regionSize);

    ACLNN_CMD(aclnnStep4BlockSparseAttention, query, key, value, packedRegions, regionCounts, kvSeqlens, numHeads,
              numKeyValueHeads, regionSize, pageSize, scaleValue, layoutQStr, layoutKvStr, numSplits,
              returnSoftmaxLseFlag, attentionOut, softmaxLse);

    return std::make_tuple(attentionOut, softmaxLse);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("step4_block_sparse_attention", &op_api::step4_block_sparse_attention, "step4_block_sparse_attention");
}
