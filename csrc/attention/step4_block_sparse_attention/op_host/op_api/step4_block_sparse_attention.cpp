/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "step4_block_sparse_attention.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Step4BlockSparseAttention);

const std::array<const aclTensor *, 2> Step4BlockSparseAttention(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *packedRegions,
    const aclTensor *regionCounts, const aclTensor *kvSeqlensOptional, int64_t numHeads, int64_t numKeyValueHeads,
    int64_t regionSize, int64_t pageSize, float scaleValue, const char *layoutQ,
    const char *layoutKv, int64_t numSplits, int64_t returnSoftmaxLse, aclOpExecutor *executor)
{
    L0_DFX(Step4BlockSparseAttention, query, key, value, packedRegions, regionCounts, kvSeqlensOptional, numHeads,
           numKeyValueHeads, regionSize, pageSize, scaleValue, layoutQ, layoutKv, numSplits,
           returnSoftmaxLse);

    auto attentionOutTensor = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseTensor = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        Step4BlockSparseAttention,
        OP_INPUT(query, key, value, packedRegions, regionCounts, kvSeqlensOptional),
        OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
        OP_ATTR(numHeads, numKeyValueHeads, regionSize, pageSize, scaleValue, layoutQ, layoutKv, numSplits,
                returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Step4BlockSparseAttention infer shape failed.");
        return {nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        Step4BlockSparseAttention,
        OP_INPUT(query, key, value, packedRegions, regionCounts, kvSeqlensOptional),
        OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
        OP_ATTR(numHeads, numKeyValueHeads, regionSize, pageSize, scaleValue, layoutQ, layoutKv, numSplits,
                returnSoftmaxLse));

    return {attentionOutTensor, softmaxLseTensor};
}

} // namespace l0op
