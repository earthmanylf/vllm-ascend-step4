/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t SOFTMAX_LSE_INDEX = 1;
static constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 8;

static constexpr uint32_t TND_DIM_T = 0;
static constexpr uint32_t TND_DIM_N = 1;
static constexpr uint32_t TND_DIM_NUM = 3;
static constexpr int32_t UNKNOWN_DIMS = -2;

static ge::graphStatus InferShapeStep4BlockSparseAttention(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    if (queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    *attentionOutShape = *queryShape;

    int64_t returnLse = 1;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *returnLsePtr = attrs->GetInt(ATTR_RETURN_SOFTMAX_LSE_INDEX);
        if (returnLsePtr != nullptr) {
            returnLse = *returnLsePtr;
        }
    }

    if (returnLse == 0) {
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    // TND query [T, N1, D] → LSE [T, N1]
    softmaxLseShape->SetDimNum(2);
    if (queryShape->GetDimNum() >= TND_DIM_NUM) {
        (*softmaxLseShape)[TND_DIM_T] = queryShape->GetDim(TND_DIM_T);
        (*softmaxLseShape)[TND_DIM_N] = queryShape->GetDim(TND_DIM_N);
    } else {
        (*softmaxLseShape)[0] = 0;
        (*softmaxLseShape)[1] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeStep4BlockSparseAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(ATTENTION_OUT_INDEX, context->GetInputDataType(QUERY_INDEX));
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Step4BlockSparseAttention)
    .InferShape(InferShapeStep4BlockSparseAttention)
    .InferDataType(InferDataTypeStep4BlockSparseAttention);

} // namespace ops
