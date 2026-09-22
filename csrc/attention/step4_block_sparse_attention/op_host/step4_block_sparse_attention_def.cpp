/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Prefill: packed_regions INT32 phys_region | (valid_tokens << 24);
 *          skip is -1 only.
 * Decode:  packed_regions INT64 start_token | (phys_region << 32);
 *          valid = clamp(kv_seqlen - start, 0, region_size).
 * BF16/FP16 QKV, arch35 only. page_size is paged block (blocksize), default 128.
 * num_splits>1 deferred.
 */

#include "register/op_def_registry.h"

namespace ops {

class Step4BlockSparseAttention : public OpDef {
public:
    explicit Step4BlockSparseAttention(const char *name) : OpDef(name)
    {
        // DataType lists are zipped by index, not a cartesian product.
        // 0: BF16 Prefill (INT32 packed), 1: BF16 Decode (INT64 packed),
        // 2: FP16 Prefill, 3: FP16 Decode.
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // KV cache TGD: [T, G, D]
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("packed_regions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("region_counts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_seqlens")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Attr("num_heads").AttrType(REQUIRED).Int(64);
        this->Attr("num_key_value_heads").AttrType(REQUIRED).Int(4);
        this->Attr("region_size").AttrType(OPTIONAL).Int(8);
        this->Attr("page_size").AttrType(OPTIONAL).Int(128);
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0);
        this->Attr("layout_q").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("TGD");
        this->Attr("num_splits").AttrType(OPTIONAL).Int(1);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Int(1);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(Step4BlockSparseAttention);

} // namespace ops
