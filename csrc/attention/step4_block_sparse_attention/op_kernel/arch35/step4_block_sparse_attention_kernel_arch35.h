/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * arch35: packed-region gather + BSA transposed high-prec QK/softmax/PV.
 * PackedT = int32_t (Prefill) or int64_t (Decode).
 */

#include "../arch35/kernel_utils.hpp"

using namespace NpuArch;
using namespace tla;

namespace BsaKernelArch35 {

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO, class PackedT>
class Step4BsaKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;
    using ElementQ = typename BlockMmadQK::ElementQ;
    using ElementK = typename BlockMmadQK::ElementK;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementQ;
    using ElementLse = typename EpilogueRescaleO::ElementLse;

    using LayoutQ = layout::ColumnMajor;
    using LayoutK = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
    static constexpr bool IS_DECODE = std::is_same<PackedT, int64_t>::value;

    __aicore__ inline Step4BsaKernelArch35() {}

    __aicore__ inline void operator()(Step4KernelParamsArch35 const &params)
    {
        __gm__ Step4BsaTilingData *tiling = reinterpret_cast<__gm__ Step4BsaTilingData *>(params.tiling);
        FetchTiling(tiling);
        CalcOnChipBufTileInfo();
        CalcUBufTileInfo();

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<PackedT> gPacked;
        gPacked.SetGlobalBuffer((__gm__ PackedT *)params.packedRegions);
        AscendC::GlobalTensor<int32_t> gCounts;
        gCounts.SetGlobalBuffer((__gm__ int32_t *)params.regionCounts);
        AscendC::GlobalTensor<int32_t> gSeqlens;
        if constexpr (IS_DECODE) {
            gSeqlens.SetGlobalBuffer((__gm__ int32_t *)params.kvSeqlens);
        }
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
        AscendC::GlobalTensor<uint32_t> gSegList;
        gSegList.SetGlobalBuffer((__gm__ uint32_t *)params.workSpace);

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
        Gemm::Block::PackedRegionRs8Tile listRing[PRE_LAUNCH + 1];
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_, uBufTileHelper_);
        EpilogueRescaleO epilogueRescaleO(resource, uBufTileHelper_);
#endif
        uint32_t groupSize = qHeads_ / kvHeads_;
        int64_t strideQ = static_cast<int64_t>(qHeads_) * embed_;
        int64_t strideKV = static_cast<int64_t>(kvHeads_) * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);

#ifdef __DAV_VEC__
        uint32_t packedAlign = ((topK_ + 32) / 32) * 32;
        if (packedAlign < 64) {
            packedAlign = 64;
        }
        uint32_t packedBytes = packedAlign * sizeof(PackedT);
        auto ubPacked =
            resource.ubBuf.template GetBufferByByte<PackedT>(uBufTileHelper_.maskStartOffset);
        auto ubGm =
            resource.ubBuf.template GetBufferByByte<int32_t>(uBufTileHelper_.maskStartOffset + packedBytes);
        auto ubList =
            resource.ubBuf.template GetBufferByByte<uint32_t>(uBufTileHelper_.maskStartOffset + packedBytes);
        uint32_t eleNumPerC0 = NpuArch::BYTE_PER_C0 / sizeof(ElementQ);
        uint64_t tokenStride = static_cast<uint64_t>(kvHeads_) * embed_;
#endif
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            uint32_t kvGroup = (qTokens_ == 0) ? 0 : (taskIdx / qTokens_);
            uint32_t qIdx = (qTokens_ == 0) ? 0 : (taskIdx - kvGroup * qTokens_);
            uint32_t packedRow = kvGroup * qTokens_ + qIdx;
            uint32_t packedRowStart = packedRow * topK_;
            uint32_t count = gCounts.GetValue(packedRow);
            int32_t kvSeqlen = 0;
            if constexpr (IS_DECODE) {
                kvSeqlen = gSeqlens.GetValue(qIdx);
            }
            uint32_t gatheredKvSeqlen =
                CompactedPackedTokenCount<IS_DECODE>(gPacked, packedRowStart, count, regionSize_, kvSeqlen);
            // GM O/LSE use real GQA groupSize; Cube/softmax/PV N is 16-aligned qBaseTile.
            uint32_t rowNum = groupSize;
            uint32_t rowNumPad = qBaseTile_;
            uint32_t rowNumRound = RoundUp(rowNumPad, 16);

            int64_t gmOffsetQ = static_cast<int64_t>(qIdx) * strideQ + static_cast<int64_t>(kvGroup) * groupSize * embed_;
            int64_t gmOffsetK = static_cast<int64_t>(kvGroup) * embed_;
            int64_t gmOffsetO = gmOffsetQ;
            int64_t gmOffsetLse = static_cast<int64_t>(qIdx) * qHeads_ + static_cast<int64_t>(kvGroup) * groupSize;

            if (gatheredKvSeqlen == 0) {
                continue;
            }

            uint32_t kvSLoopNum = CeilDiv(gatheredKvSeqlen, kvBaseTile_);
            uint32_t kvSTileSizeAct = kvBaseTile_;
            uint32_t rowListBase = taskIdx * maxKvTiles_ * Gemm::Block::STEP4_SEG_LIST_WORDS;
            Arch::CrossCoreFlag listReadyFlag(Gemm::Block::STEP4_LIST_READY_FLAG0);
#ifdef __DAV_VEC__
            uint32_t packedLeft = 0;
            uint32_t packedSlot = 0;
            uint32_t packedCompact = 0;
            Gemm::Block::PackedRegionLoadPackedRowToUb(gPacked, packedRow, totalTaskNum_, topK_, packedAlign, count,
                                                       ubPacked, packedLeft);
            if (kvSLoopNum > 0) {
                Gemm::Block::PackedRegionDecodeOneTileSplitToGm<IS_DECODE>(
                    ubPacked, ubGm, ubList, count, regionSize_, pageSize_, kvBaseTile_, gatheredKvSeqlen, 0, kvSeqlen,
                    packedCompact, packedSlot, tokenStride, eleNumPerC0, packedLeft, mm1L1TileN_,
                    (__gm__ uint32_t *)params.workSpace, rowListBase);
                Epilogue::Block::SetCrossCoreSync<4, PIPE_MTE3>(listReadyFlag);
            }
            // TND one token's GQA group: groupSize heads are contiguous, stride = D (not N1*D).
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(rowNum, embed_);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
            auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(rowNum, 1);
            auto gmLseTensorTla = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayoutTla, Arch::PositionGM{});
#endif
#ifdef __DAV_CUBE__
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(embed_, rowNum);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);

            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kvTokens_, strideKV);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});
            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvTokens_, strideKV);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetK], gmVLayoutTla, Arch::PositionGM{});
#endif
            for (uint32_t gatheredKvSTileIdx = 0; gatheredKvSTileIdx < kvSLoopNum + PRE_LAUNCH; gatheredKvSTileIdx++) {
                if (gatheredKvSTileIdx < kvSLoopNum) {
                    kvSTileSizeAct = (gatheredKvSTileIdx == kvSLoopNum - 1) ?
                                         (gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_) :
                                         kvBaseTile_;
                    GemmCoord actualBlockShapeQK{rowNumPad, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = gatheredKvSTileIdx % UB_S_OTMP_BUF_STAGES;
                    uint32_t sShapeRow = RoundUp(kvSTileSizeAct, 16);
                    uint32_t sShapeCol = uBufTileHelper_.qBaseTilePerSubCore;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(sShapeRow, sShapeCol);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    Arch::CrossCoreFlag mm1ToSmFlag(ubSBufId);
                    uint32_t listWordOff = rowListBase + gatheredKvSTileIdx * Gemm::Block::STEP4_SEG_LIST_WORDS;
#ifdef __DAV_VEC__
                    if (gatheredKvSTileIdx + 1 < kvSLoopNum) {
                        uint32_t listWordOffNext =
                            rowListBase + (gatheredKvSTileIdx + 1) * Gemm::Block::STEP4_SEG_LIST_WORDS;
                        Gemm::Block::PackedRegionDecodeOneTileSplitToGm<IS_DECODE>(
                            ubPacked, ubGm, ubList, count, regionSize_, pageSize_, kvBaseTile_, gatheredKvSeqlen,
                            gatheredKvSTileIdx + 1, kvSeqlen, packedCompact, packedSlot, tokenStride, eleNumPerC0,
                            packedLeft, mm1L1TileN_, (__gm__ uint32_t *)params.workSpace, listWordOffNext);
                        Epilogue::Block::SetCrossCoreSync<4, PIPE_MTE3>(listReadyFlag);
                    }
#endif
#ifdef __DAV_CUBE__
                    Gemm::Block::WaitCrossCoreSync<4, PIPE_S>(listReadyFlag);
                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdx, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdx, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, true);
                    blockMmadQK(gmKTensorTla, ubSTensorTla, gPacked, packedRowStart, count, regionSize_,
                                pageSize_, actualBlockShapeQK, gatheredKvSTileIdx, kvBaseTile_, prefixSumL0AStages,
                                prefixSumL0BStages, mm1ToSmFlag, kvSeqlen, gSegList, listWordOff,
                                &listRing[gatheredKvSTileIdx % (PRE_LAUNCH + 1)]);
                    if (gatheredKvSTileIdx == kvSLoopNum - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    uint32_t l1PBufId = gatheredKvSTileIdx % pL1BufNum_;
                    Arch::CrossCoreFlag smToMm2Flag(l1PBufId + UB_S_OTMP_BUF_STAGES);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::zN>(rowNumPad, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (gatheredKvSTileIdx == 0), ubSBufId,
                                          l1PBufId, mm1ToSmFlag, smToMm2Flag);
#endif
                }
                if (gatheredKvSTileIdx >= PRE_LAUNCH) {
                    uint32_t gatheredKvSTileIdxDe = gatheredKvSTileIdx - PRE_LAUNCH;
                    kvSTileSizeAct = (gatheredKvSTileIdxDe == kvSLoopNum - 1) ?
                                         (gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_) :
                                         kvBaseTile_;
                    GemmCoord actualBlockShapePV{rowNumPad, embed_, kvSTileSizeAct};
                    GemmCoord actualBlockShapeO{rowNum, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = gatheredKvSTileIdxDe % UB_S_OTMP_BUF_STAGES;
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = gatheredKvSTileIdxDe % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    Arch::CrossCoreFlag smToMm2Flag(l1PBufId + UB_S_OTMP_BUF_STAGES);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdxDe, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdxDe, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, false);
                    uint32_t listWordOffDe = rowListBase + gatheredKvSTileIdxDe * Gemm::Block::STEP4_SEG_LIST_WORDS;
                    blockMmadPV(gmVTensorTla, ubOTmpTensorTla, gPacked, packedRowStart, count, regionSize_,
                                pageSize_, actualBlockShapePV, gatheredKvSTileIdxDe, kvBaseTile_, prefixSumL0AStages,
                                prefixSumL0BStages, smToMm2Flag, mm2ToReFlag, kvSeqlen, gSegList, listWordOffDe,
                                mm1L1TileN_, &listRing[gatheredKvSTileIdxDe % (PRE_LAUNCH + 1)]);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = gatheredKvSTileIdxDe % (PRE_LAUNCH + 1);
                    epilogueRescaleO(gmOTensorTla, gmLseTensorTla, actualBlockShapeO, curTileMod, gatheredKvSTileIdxDe,
                                     (gatheredKvSTileIdxDe == 0), (gatheredKvSTileIdxDe == kvSLoopNum - 1),
                                     mm2ToReFlag);
#endif
                }
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
    }

private:
    __aicore__ inline void FetchTiling(__gm__ Step4BsaTilingData *tiling)
    {
        qTokens_ = tiling->qTokens;
        kvTokens_ = tiling->kvTokens;
        qHeads_ = tiling->numHeads;
        kvHeads_ = tiling->kvHeads;
        embed_ = tiling->embeddingSize;
        regionSize_ = tiling->regionSize;
        pageSize_ = tiling->pageSize;
        topK_ = tiling->topK;
        qBaseTile_ = tiling->qBaseTile;
        kvBaseTile_ = tiling->kvBaseTile;
        totalTaskNum_ = tiling->totalTaskNum;
        scaleValue_ = tiling->scaleValue;
        mm1L1TileM_ = tiling->mm1L1TileM;
        mm1L1TileN_ = tiling->mm1L1TileN;
        mm1L1TileKLeft_ = tiling->mm1L1TileKLeft;
        mm1L1TileKRight_ = tiling->mm1L1TileKRight;
        mm2L1TileM_ = tiling->mm2L1TileM;
        mm2L1TileN_ = tiling->mm2L1TileN;
        mm2L1TileKLeft_ = tiling->mm2L1TileKLeft;
        mm2L1TileKRight_ = tiling->mm2L1TileKRight;
        qL1BufNum_ = tiling->qL1BufNum;
        kL1BufNum_ = tiling->kL1BufNum;
        vL1BufNum_ = tiling->vL1BufNum;
        pL1BufNum_ = tiling->pL1BufNum;
        maxKvTiles_ = tiling->maxKvTiles;
        if (maxKvTiles_ == 0) {
            maxKvTiles_ = 1;
        }
    }

    __aicore__ inline void CalcOnChipBufTileInfo()
    {
        mm1L1TileHelper_ = Gemm::Block::Mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
                                                         qL1BufNum_, kL1BufNum_);
        mm2L1TileHelper_ = Gemm::Block::Mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
                                                         pL1BufNum_, vL1BufNum_);
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                          mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        mm1L0ATotalStages_ = CeilDiv(kvBaseTile_, BlockMmadQK::L0_TILE_M) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
        mm1L0BTotalStages_ = CeilDiv(qBaseTile_, BlockMmadQK::L0_TILE_N) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
        mm2L0ATotalStages_ = CeilDiv(qBaseTile_, BlockMmadPV::L0_TILE_M) * CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K);
        mm2L0BTotalStages_ = CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K) * CeilDiv(embed_, BlockMmadPV::L0_TILE_N);
    }

    __aicore__ inline void CalcUBufTileInfo()
    {
        // 1:2 MIX: softmax splits N with RoundUp(q,32)/2. groupSize=16 keeps 16 on AIV0 and idles AIV1.
        // High-prec VF StoreAlign(..., pregFull) writes 64 floats; keep stats UB at 64, S/O at qBaseTile.
        // P is 64 Q-cols (HIGH_PREC_P_NZ_COLS): nd2nz reads lpUb[blockStride*64], not q*kv.
        uint32_t qBaseTilePerSubCore = qBaseTile_;
        uint32_t qStatTilePerSubCore = 64;
        uint32_t kvBaseTilePerSubCore = RoundUp(kvBaseTile_, 32);
        uint32_t embedPerSubCore = RoundUp(embed_, 32);
        uint32_t sStartOffset = 0;
        uint32_t pStartOffset =
            sStartOffset + qBaseTilePerSubCore * kvBaseTilePerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES;
        uint32_t loStartOffset =
            pStartOffset +
            Epilogue::Block::HighPrecPStageElems(kvBaseTilePerSubCore) * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES;
        uint32_t goStartOffset =
            loStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp) * UB_S_OTMP_BUF_STAGES;
        uint32_t lmStartOffset = goStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp);
        uint32_t gmStartOffset = lmStartOffset + qStatTilePerSubCore * sizeof(float);
        uint32_t dmStartOffset = gmStartOffset + qStatTilePerSubCore * sizeof(float);
        uint32_t llStartOffset = dmStartOffset + qStatTilePerSubCore * (PRE_LAUNCH + 1) * sizeof(float);
        uint32_t glStartOffset = llStartOffset + qStatTilePerSubCore * sizeof(float);
        uint32_t lseStartOffset = glStartOffset + qStatTilePerSubCore * sizeof(float);
        uint32_t maskStartOffset = lseStartOffset + qStatTilePerSubCore * (32 / sizeof(float)) * sizeof(float);
        uBufTileHelper_ = Epilogue::Block::UBufTileHelper(
            qBaseTilePerSubCore, qStatTilePerSubCore, kvBaseTilePerSubCore, embedPerSubCore, sStartOffset, pStartOffset,
            loStartOffset, goStartOffset, lmStartOffset, gmStartOffset, dmStartOffset, llStartOffset, glStartOffset,
            lseStartOffset, maskStartOffset);
    }

    __aicore__ inline uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(uint32_t gatheredKvSTileIdx,
                                                                  uint32_t singleMm1L0Stages,
                                                                  uint32_t singleMm2L0Stages, uint32_t kvSLoopNum,
                                                                  bool isCurPhaseMm1)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseMm1) {
            prefixSumStages =
                (gatheredKvSTileIdx <= PRE_LAUNCH) ?
                    gatheredKvSTileIdx * singleMm1L0Stages :
                    gatheredKvSTileIdx * singleMm1L0Stages + (gatheredKvSTileIdx - PRE_LAUNCH) * singleMm2L0Stages;
        } else {
            prefixSumStages =
                (gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH) ?
                    (gatheredKvSTileIdx + 1 + PRE_LAUNCH) * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages :
                    kvSLoopNum * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages;
        }
        return prefixSumStages;
    }

    __aicore__ inline void InitCrossCoreDstBuf(AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
                                               AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
                                               AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                uBufTileHelper_.sStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.kvBaseTilePerSubCore * sizeof(ElementS) * i);
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper_.loStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.embedPerSubCore * sizeof(ElementOTmp) * i);
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(1);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(5);
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(6);
        }
#endif
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    Arch::Resource<ArchTag> resource;
    uint32_t qTokens_;
    uint32_t kvTokens_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t regionSize_;
    uint32_t pageSize_;
    uint32_t topK_;
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    uint32_t mm1L1TileM_;
    uint32_t mm1L1TileN_;
    uint32_t mm1L1TileKLeft_;
    uint32_t mm1L1TileKRight_;
    uint32_t mm2L1TileM_;
    uint32_t mm2L1TileN_;
    uint32_t mm2L1TileKLeft_;
    uint32_t mm2L1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;
    uint32_t maxKvTiles_;
    uint32_t mm1L0ATotalStages_;
    uint32_t mm1L0BTotalStages_;
    uint32_t mm2L0ATotalStages_;
    uint32_t mm2L0BTotalStages_;
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
    Epilogue::Block::UBufTileHelper uBufTileHelper_;
};

} // namespace BsaKernelArch35
