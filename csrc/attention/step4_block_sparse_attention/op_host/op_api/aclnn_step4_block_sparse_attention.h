/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_STEP4_BLOCK_SPARSE_ATTENTION_H_
#define ACLNN_STEP4_BLOCK_SPARSE_ATTENTION_H_

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnStep4BlockSparseAttention 第一段接口，根据计算流程计算 workspace 大小。
 * @domain aclnn_ops_infer
 *
 * Prefill 与 Decode 共用本接口，由 packedRegions dtype 区分：
 * INT32 Prefill，INT64 Decode。
 *
 * query            TND：[T, N1, D]。
 * key / value      TGD：[T_kv, G, D]，直接从 KV cache 读取。
 * packedRegions    group-major [G * T, K]，K=topk+1。Decode 时 T=B。
 *                  Prefill INT32：phys_region | (valid_tokens << 24)，valid 1–255；
 *                  无效只填 -1（0xFFFFFFFF），不要把任意负数当 skip。
 *                  Decode  INT64：start_token | (phys_region << 32)，无效填 0；
 *                           可见长度 clamp(kv_seqlen - start, 0, regionSize)。
 *                           packed==0 是 region-0 合法编码，不要当 skip。
 * regionCounts     INT32 [G * T]，以它为准，不要扫到 K。
 * kvSeqlensOptional Decode 必传 INT32 [B]，含当前 token；Prefill 传 nullptr。
 * pageSize         paged block（blocksize），默认 128，须整除 regionSize。
 *                  TGD 扁平 cache 上等价于 phys * regionSize；PA + block_table 未支持。
 * numSplits        初版仅 1；>1 的 split-KV merge 后续再做。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnStep4BlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *packedRegions,
    const aclTensor *regionCounts, const aclTensor *kvSeqlensOptional, int64_t numHeads, int64_t numKeyValueHeads,
    int64_t regionSize, int64_t pageSize, double scaleValue, char *layoutQ, char *layoutKv,
    int64_t numSplits, int64_t returnSoftmaxLse, aclTensor *attentionOut, aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnStep4BlockSparseAttention 第二段接口，执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnStep4BlockSparseAttention(void *workspace,
                                                                                  uint64_t workspaceSize,
                                                                                  aclOpExecutor *executor,
                                                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_STEP4_BLOCK_SPARSE_ATTENTION_H_
