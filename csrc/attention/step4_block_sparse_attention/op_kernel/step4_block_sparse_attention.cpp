/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/step4_block_sparse_attention_kernel_arch35.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 310)
using namespace BsaKernelArch35;

template <class InDtype, class SMDtype, class REDtype, class PackedT>
__global__ __aicore__ void Step4BsaIntf(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR packedRegions, GM_ADDR regionCounts,
                                        GM_ADDR kvSeqlens, GM_ADDR o, GM_ADDR softmaxLse, GM_ADDR workspace,
                                        GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;

    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::nZ;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK<true>;
    using TileCopyQK =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, layout::RowMajor, ElementK, layout::ColumnMajor, ElementS,
                                          LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa<true>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;

    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV<true>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;

    using DispatchPolicyRescaleO =
        Epilogue::EpilogueAtlasA5BsaRescaleO<Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::BNS1, true>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using Kernel =
        Step4BsaKernelArch35<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, PackedT>;
    Step4KernelParamsArch35 params{q, k, v, packedRegions, regionCounts, kvSeqlens, o, softmaxLse, workspace, tiling};
    Kernel kernel;
    kernel(params);
}
#endif

extern "C" __global__ __aicore__ void step4_block_sparse_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *packedRegions,
    __gm__ uint8_t *regionCounts, __gm__ uint8_t *kvSeqlens, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310)
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TILING_KEY_IS(0);
    TILING_KEY_IS(1);
    TILING_KEY_IS(2);
    TILING_KEY_IS(3);
    if (TILING_KEY_IS(0)) {
        Step4BsaIntf<bfloat16_t, float, float, int32_t>(query, key, value, packedRegions, regionCounts, kvSeqlens,
                                                            attentionOut, softmaxLse, user, tiling);
    } else if (TILING_KEY_IS(1)) {
        Step4BsaIntf<bfloat16_t, float, float, int64_t>(query, key, value, packedRegions, regionCounts, kvSeqlens,
                                                            attentionOut, softmaxLse, user, tiling);
    } else if (TILING_KEY_IS(2)) {
        Step4BsaIntf<half, float, float, int32_t>(query, key, value, packedRegions, regionCounts, kvSeqlens,
                                                      attentionOut, softmaxLse, user, tiling);
    } else if (TILING_KEY_IS(3)) {
        Step4BsaIntf<half, float, float, int64_t>(query, key, value, packedRegions, regionCounts, kvSeqlens,
                                                      attentionOut, softmaxLse, user, tiling);
    }
#else
    (void)kvSeqlens;
#endif
}
