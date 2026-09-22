/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_step4_block_sparse_attention.h"

#include "step4_block_sparse_attention.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr int64_t LSE_OUT = 1;

static const aclTensor *MaybeContiguous(const aclTensor *tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    return l0op::Contiguous(tensor, executor);
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnStep4BlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *packedRegions,
    const aclTensor *regionCounts, const aclTensor *kvSeqlensOptional, int64_t numHeads, int64_t numKeyValueHeads,
    int64_t regionSize, int64_t pageSize, double scaleValue, char *layoutQ, char *layoutKv,
    int64_t numSplits, int64_t returnSoftmaxLse, aclTensor *attentionOut, aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(packedRegions != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(regionCounts != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (packedRegions->GetDataType() == DataType::DT_INT64 && kvSeqlensOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "kvSeqlens is required when packedRegions is INT64 (Decode).");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const DataType queryDtype = query->GetDataType();
    if (queryDtype != DataType::DT_BF16 && queryDtype != DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query dtype must be BF16 or FP16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (queryDtype != key->GetDataType() || queryDtype != value->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query/key/value must share the same dtype.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutQ == nullptr || layoutKv == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "layoutQ/layoutKv must not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnStep4BlockSparseAttention,
                   DFX_IN(query, key, value, packedRegions, regionCounts, kvSeqlensOptional, numHeads, numKeyValueHeads,
                          regionSize, pageSize, scaleValue, layoutQ, layoutKv, numSplits, returnSoftmaxLse),
                   DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    query = l0op::Contiguous(query, executorImpl);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Contiguous(key, executorImpl);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Contiguous(value, executorImpl);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    packedRegions = l0op::Contiguous(packedRegions, executorImpl);
    CHECK_RET(packedRegions != nullptr, ACLNN_ERR_INNER_NULLPTR);
    regionCounts = l0op::Contiguous(regionCounts, executorImpl);
    CHECK_RET(regionCounts != nullptr, ACLNN_ERR_INNER_NULLPTR);
    kvSeqlensOptional = MaybeContiguous(kvSeqlensOptional, executorImpl);

    const std::string layoutQStr = op::ToString(layoutQ).GetString();
    const std::string layoutKvStr = op::ToString(layoutKv).GetString();

    auto outputs = l0op::Step4BlockSparseAttention(
        query, key, value, packedRegions, regionCounts, kvSeqlensOptional, numHeads, numKeyValueHeads, regionSize,
        pageSize, static_cast<float>(scaleValue), layoutQStr.c_str(), layoutKvStr.c_str(), numSplits,
        returnSoftmaxLse, executorImpl);
    CHECK_RET(outputs[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (returnSoftmaxLse == LSE_OUT) {
        CHECK_RET(softmaxLseOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(outputs[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyLseResult = l0op::ViewCopy(outputs[1], softmaxLseOptional, executorImpl);
        CHECK_RET(viewCopyLseResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnStep4BlockSparseAttention(void *workspace,
                                                                                  uint64_t workspaceSize,
                                                                                  aclOpExecutor *executor,
                                                                                  aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStep4BlockSparseAttention);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
