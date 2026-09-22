/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef STEP4_PACKED_REGION_GATHER_ARCH35_HPP
#define STEP4_PACKED_REGION_GATHER_ARCH35_HPP

#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"

namespace NpuArch::Gemm::Block {

static constexpr uint32_t STEP4_GATHER_MAX_SEG = 32;
// One 256-tile GM slot holds two QK L1 halves (mm1L1TileN=128). Cube issues each half
// directly from GM (no PackedSegList spill). Host workspace stays 200 words / 256-tile.
// Half stride is 104 words / 416B = 13 * 32B: each half owns whole 32B GM blocks, so the two
// AIVs never write the same block. 100 words (400B) let AIV0's tail and AIV1's header share one.
static constexpr uint32_t STEP4_SEG_LIST_WORDS = 208;
static constexpr uint32_t STEP4_SEG_HALF_WORDS = 104;
static constexpr uint32_t STEP4_SEG_LIST_BYTES = STEP4_SEG_LIST_WORDS * sizeof(uint32_t);
// Mode-4 is set_intra_block, not FFTS. 11-13/14/28-29 are SyncAll (14=SYNC_AIV_ONLY_ALL).
static constexpr uint16_t STEP4_LIST_READY_FLAG0 = 15;
static constexpr uint32_t STEP4_SEG_META_NDNUM_MASK = 0xFFu;
static constexpr uint32_t STEP4_RS8 = 8;
static constexpr uint32_t STEP4_RS8_HDR_TAIL_SHIFT = 16;
static constexpr uint32_t STEP4_RS8_MAX_PAIR = 16;
static constexpr uint32_t STEP4_RS8_MAX_TAIL = 16;

// One rs=8 half (100 GM words decoded). QK fills it; PV reuses via a PRE_LAUNCH+1 ring.
struct PackedRegionRs8Half {
    uint32_t nPair;
    uint32_t nTail;
    uint32_t pairGm[STEP4_RS8_MAX_PAIR];
    uint32_t pairLo[STEP4_RS8_MAX_PAIR];
    uint32_t pairHi[STEP4_RS8_MAX_PAIR];
    uint32_t tailGm[STEP4_RS8_MAX_TAIL];
};

struct PackedRegionRs8Tile {
    uint32_t halfValid[2];
    PackedRegionRs8Half half[2];
};

static constexpr uint32_t STEP4_PACKED_REGION_ID_MASK = 0x00FFFFFFu;
static constexpr uint32_t STEP4_PACKED_VALID_SHIFT = 24;
static constexpr uint32_t STEP4_PACKED_PREFILL_SKIP = 0xFFFFFFFFu;

template <bool IS_DECODE, class PackedT>
__aicore__ inline void ParsePackedSlot(PackedT meta, uint32_t regionSize, int32_t kvSeqlen, uint32_t &phys,
                                       uint32_t &valid)
{
    if constexpr (IS_DECODE) {
        uint64_t word = static_cast<uint64_t>(meta);
        uint32_t start = static_cast<uint32_t>(word);
        phys = static_cast<uint32_t>(word >> 32);
        int64_t vis = static_cast<int64_t>(kvSeqlen) - static_cast<int64_t>(start);
        if (vis < 0) {
            vis = 0;
        }
        if (vis > static_cast<int64_t>(regionSize)) {
            vis = static_cast<int64_t>(regionSize);
        }
        valid = static_cast<uint32_t>(vis);
        (void)start;
    } else {
        // Prefill valid sits in bits [31:24]. Skip is only the sentinel 0xFFFFFFFF (-1);
        // do not treat any negative INT32 as skip (high valid bits share the sign bit).
        uint32_t word = static_cast<uint32_t>(meta);
        if (word == STEP4_PACKED_PREFILL_SKIP) {
            phys = 0;
            valid = 0;
            return;
        }
        phys = word & STEP4_PACKED_REGION_ID_MASK;
        valid = word >> STEP4_PACKED_VALID_SHIFT;
    }
}

// TGD GM row of region `phys`'s first token. pageSize is the paged block (blocksize);
// regionSize stays 8. On a cache flattened in region-id order this equals phys * regionSize:
//   (phys / rpp) * pageSize + (phys % rpp) * regionSize, rpp = pageSize / regionSize.
__aicore__ inline uint32_t PackedRegionGmStart(uint32_t phys, uint32_t regionSize, uint32_t pageSize)
{
    if (regionSize == 0) {
        return 0;
    }
    if (pageSize == 0 || (pageSize % regionSize) != 0) {
        return phys * regionSize;
    }
    uint32_t rpp = pageSize / regionSize;
    return (phys / rpp) * pageSize + (phys % rpp) * regionSize;
}

// One packed slot clipped to [gatheredStart, targetEnd). slotRef is the first not-fully-consumed
// slot; compactPosRef is that slot's compact start. A tile that ends mid-slot leaves both so the
// next tile resumes via copyStartInSlot.
template <bool IS_DECODE, class PackedT>
__aicore__ inline uint32_t PackedNextSlot(AscendC::GlobalTensor<PackedT> &gPacked, uint32_t packedRowStart,
                                          uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                          uint32_t gatheredStart, uint32_t targetEnd, int32_t kvSeqlen,
                                          uint32_t &compactPosRef, uint32_t &slotRef, uint32_t &segOriStart)
{
    while (slotRef < packedCount) {
        PackedT meta = gPacked.GetValue(packedRowStart + slotRef);
        uint32_t phys = 0;
        uint32_t valid = 0;
        ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, valid);
        if (valid == 0) {
            ++slotRef;
            continue;
        }
        uint32_t slotStart = compactPosRef;
        compactPosRef += valid;
        if (compactPosRef <= gatheredStart) {
            ++slotRef;
            continue;
        }
        if (slotStart >= targetEnd) {
            compactPosRef = slotStart;
            break;
        }
        uint32_t copyStartInSlot = (gatheredStart > slotStart) ? (gatheredStart - slotStart) : 0;
        uint32_t copyEndInSlot = valid;
        uint32_t slotCap = targetEnd - slotStart;
        if (slotCap < copyEndInSlot) {
            copyEndInSlot = slotCap;
        }
        uint32_t segLen = copyEndInSlot - copyStartInSlot;
        if (segLen == 0) {
            compactPosRef = slotStart;
            ++slotRef;
            continue;
        }
        segOriStart = PackedRegionGmStart(phys, regionSize, pageSize) + copyStartInSlot;
        if (copyEndInSlot < valid) {
            compactPosRef = slotStart;
            return segLen;
        }
        ++slotRef;
        return segLen;
    }
    return 0;
}

// Same walk as PackedNextSlot, but packed already lives in UB (mask2idx-style MTE2 burst).
template <bool IS_DECODE, class PackedT>
__aicore__ inline uint32_t PackedNextSlotUb(AscendC::LocalTensor<PackedT> ubPacked, AscendC::LocalTensor<int32_t> ubGm,
                                            bool hasGm, uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                            uint32_t gatheredStart, uint32_t targetEnd, int32_t kvSeqlen,
                                            uint32_t &compactPosRef, uint32_t &slotRef, uint32_t &segOriStart,
                                            uint32_t ubOff)
{
    while (slotRef < packedCount) {
        PackedT meta = ubPacked.GetValue(slotRef + ubOff);
        uint32_t phys = 0;
        uint32_t valid = 0;
        ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, valid);
        if (valid == 0) {
            ++slotRef;
            continue;
        }
        uint32_t slotStart = compactPosRef;
        compactPosRef += valid;
        if (compactPosRef <= gatheredStart) {
            ++slotRef;
            continue;
        }
        if (slotStart >= targetEnd) {
            compactPosRef = slotStart;
            break;
        }
        uint32_t copyStartInSlot = (gatheredStart > slotStart) ? (gatheredStart - slotStart) : 0;
        uint32_t copyEndInSlot = valid;
        uint32_t slotCap = targetEnd - slotStart;
        if (slotCap < copyEndInSlot) {
            copyEndInSlot = slotCap;
        }
        uint32_t segLen = copyEndInSlot - copyStartInSlot;
        if (segLen == 0) {
            compactPosRef = slotStart;
            ++slotRef;
            continue;
        }
        uint32_t gmBase = hasGm ? static_cast<uint32_t>(ubGm.GetValue(slotRef + ubOff)) :
                                  PackedRegionGmStart(phys, regionSize, pageSize);
        segOriStart = gmBase + copyStartInSlot;
        if (copyEndInSlot < valid) {
            compactPosRef = slotStart;
            return segLen;
        }
        ++slotRef;
        return segLen;
    }
    return 0;
}
__aicore__ inline uint32_t PackedSegStoreWords(uint32_t ndNum)
{
    return (ndNum == 2) ? 4u : 2u;
}

__aicore__ inline uint32_t PackedSegPackMeta(uint32_t nValue, uint32_t ndNum)
{
    return (nValue << 8) | (ndNum & STEP4_SEG_META_NDNUM_MASK);
}

__aicore__ inline void PackedSegUnpackMeta(uint32_t meta, uint32_t &nValue, uint32_t &ndNum)
{
    ndNum = meta & STEP4_SEG_META_NDNUM_MASK;
    nValue = meta >> 8;
}

template <class ListBuf>
__aicore__ inline uint32_t PackedSegWriteRecordAt(ListBuf list, uint32_t base, uint32_t rel, uint32_t gm,
                                                  uint32_t nValue, uint32_t ndNum, uint64_t srcStride,
                                                  uint32_t relLimit = STEP4_SEG_LIST_WORDS)
{
    uint32_t need = PackedSegStoreWords(ndNum);
    if (rel + need > relLimit) {
        return rel;
    }
    list.SetValue(base + rel, gm);
    list.SetValue(base + rel + 1, PackedSegPackMeta(nValue, ndNum));
    if (ndNum == 2) {
        list.SetValue(base + rel + 2, static_cast<uint32_t>(srcStride));
        list.SetValue(base + rel + 3, static_cast<uint32_t>(srcStride >> 32));
    }
    return rel + need;
}

// Caller must hold MTE3_S EVENT_ID0 before writing ubList; the wait cannot live here, the
// scalar writes have already happened by then.
__aicore__ inline void PackedRegionStoreUbListHalf(AscendC::LocalTensor<uint32_t> ubList, __gm__ uint32_t *listPtr,
                                                   uint32_t wordOff)
{
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::GlobalTensor<uint32_t> gTile;
    gTile.SetGlobalBuffer(listPtr + wordOff);
    AscendC::DataCopy(gTile, ubList, STEP4_SEG_HALF_WORDS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// rs=8 pair/tail list: skip -1, each live slot is a full region. 0=end, 1=gm, 2=bad valid.
template <class PackedT>
__aicore__ inline uint32_t PackedPrefillRs8NextLiveGm(AscendC::LocalTensor<PackedT> ubPacked, uint32_t packedCount,
                                                    uint32_t regionSize, uint32_t pageSize, uint32_t ubOff,
                                                    uint32_t &slot, uint32_t &compactPos, uint32_t &gm)
{
    while (slot < packedCount) {
        uint32_t word = static_cast<uint32_t>(ubPacked.GetValue(slot + ubOff));
        ++slot;
        if (word == STEP4_PACKED_PREFILL_SKIP) {
            continue;
        }
        uint32_t valid = word >> STEP4_PACKED_VALID_SHIFT;
        if (valid == 0) {
            continue;
        }
        if (valid != STEP4_RS8) {
            return 2;
        }
        uint32_t phys = word & STEP4_PACKED_REGION_ID_MASK;
        compactPos += valid;
        gm = (pageSize != 0 && (pageSize % regionSize) == 0) ? (phys * regionSize) :
                                                               PackedRegionGmStart(phys, regionSize, pageSize);
        return 1;
    }
    return 0;
}

// Other AIV's 128-row half is live tokens, not -1. Jump slot += tok/8; do not GetValue-walk.
template <class PackedT>
__aicore__ inline bool PackedPrefillRs8SkipTo(AscendC::LocalTensor<PackedT> ubPacked, uint32_t packedCount,
                                            uint32_t regionSize, uint32_t pageSize, uint32_t ubOff,
                                            uint32_t gatheredStart, uint32_t &slot, uint32_t &compactPos)
{
    if (compactPos > gatheredStart) {
        return false;
    }
    if (compactPos == gatheredStart) {
        return true;
    }
    uint32_t tokNeed = gatheredStart - compactPos;
    if (regionSize == STEP4_RS8 && (tokNeed % STEP4_RS8) == 0) {
        uint32_t liveNeed = tokNeed / STEP4_RS8;
        if (slot + liveNeed <= packedCount) {
            slot += liveNeed;
            compactPos = gatheredStart;
            return true;
        }
    }
    while (compactPos < gatheredStart) {
        uint32_t skipGm = 0;
        uint32_t st = PackedPrefillRs8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, slot, compactPos,
                                              skipGm);
        (void)skipGm;
        if (st != 1) {
            return false;
        }
    }
    return compactPos == gatheredStart;
}

template <bool SUPPORT_PAIR, class PackedT>
__aicore__ inline void PackedRegionRs8ToUbList(AscendC::LocalTensor<PackedT> ubPacked,
                                                          AscendC::LocalTensor<int32_t> ubGm, bool hasGm,
                                                          uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                                          uint32_t gatheredStart, uint32_t gatheredLen,
                                                          uint32_t &compactPos, uint32_t &slot, uint64_t tokenStride,
                                                          uint32_t eleNumPerC0, uint32_t ubOff,
                                                          AscendC::LocalTensor<uint32_t> ubList, uint32_t listBase,
                                                          uint32_t relLimit)
{
    (void)ubGm;
    (void)hasGm;
    uint32_t nPair = 0;
    uint32_t nTail = 0;
    uint32_t rel = 1;
    uint32_t dealtLenAccum = 0;
    bool ok = PackedPrefillRs8SkipTo(ubPacked, packedCount, regionSize, pageSize, ubOff, gatheredStart, slot, compactPos);
    while (ok && slot < packedCount && dealtLenAccum < gatheredLen) {
        uint32_t gm0 = 0;
        uint32_t st0 = PackedPrefillRs8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, slot, compactPos,
                                               gm0);
        if (st0 != 1) {
            ok = (st0 == 0 && dealtLenAccum == gatheredLen);
            break;
        }
        bool storedPair = false;
        if constexpr (SUPPORT_PAIR) {
            if (dealtLenAccum + 16u <= gatheredLen && rel + 3u <= relLimit) {
                uint32_t compactPos2 = compactPos;
                uint32_t slot2 = slot;
                uint32_t gm1 = 0;
                uint32_t st1 = PackedPrefillRs8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, slot2,
                                                       compactPos2, gm1);
                if (st1 == 1 && gm1 != gm0) {
                    uint32_t gmLo = (gm0 < gm1) ? gm0 : gm1;
                    uint32_t gmHi = (gm0 < gm1) ? gm1 : gm0;
                    uint64_t stride = static_cast<uint64_t>(gmHi - gmLo) * tokenStride;
                    if (stride % eleNumPerC0 == 0) {
                        ubList.SetValue(listBase + rel, gmLo);
                        ubList.SetValue(listBase + rel + 1, static_cast<uint32_t>(stride));
                        ubList.SetValue(listBase + rel + 2, static_cast<uint32_t>(stride >> 32));
                        rel += 3;
                        compactPos = compactPos2;
                        slot = slot2;
                        dealtLenAccum += 16;
                        nPair += 1;
                        storedPair = true;
                    }
                }
            }
        }
        if (!storedPair) {
            if (dealtLenAccum + STEP4_RS8 < gatheredLen || rel + 1u > relLimit) {
                ok = false;
                break;
            }
            ubList.SetValue(listBase + rel, gm0);
            rel += 1;
            dealtLenAccum += STEP4_RS8;
            nTail += 1;
        }
    }
    uint32_t hdr = nPair | (nTail << STEP4_RS8_HDR_TAIL_SHIFT);
    ubList.SetValue(listBase, (ok && dealtLenAccum == gatheredLen) ? hdr : 0);
}

// Decode rs=8 pair/tail list: INT64 start|(phys<<32), valid=clamp(kv_seqlen-start,0,8).
// Design keeps history full-8; only the current region may be partial (returns 2 → Gather).
// Meta is compacted (no -1 holes), so SkipTo may slot += tok/8.
template <class PackedT>
__aicore__ inline uint32_t PackedDecodeRs8NextLiveGm(AscendC::LocalTensor<PackedT> ubPacked, uint32_t packedCount,
                                                          uint32_t regionSize, uint32_t pageSize, uint32_t ubOff,
                                                          int32_t kvSeqlen, uint32_t &slot, uint32_t &compactPos,
                                                          uint32_t &gm)
{
    while (slot < packedCount) {
        PackedT meta = ubPacked.GetValue(slot + ubOff);
        ++slot;
        uint32_t phys = 0;
        uint32_t valid = 0;
        ParsePackedSlot<true>(meta, regionSize, kvSeqlen, phys, valid);
        if (valid == 0) {
            continue;
        }
        if (valid != STEP4_RS8) {
            return 2;
        }
        compactPos += valid;
        gm = (pageSize != 0 && (pageSize % regionSize) == 0) ? (phys * regionSize) :
                                                               PackedRegionGmStart(phys, regionSize, pageSize);
        return 1;
    }
    return 0;
}

template <class PackedT>
__aicore__ inline bool PackedDecodeRs8SkipTo(AscendC::LocalTensor<PackedT> ubPacked, uint32_t packedCount,
                                                  uint32_t regionSize, uint32_t pageSize, uint32_t ubOff,
                                                  int32_t kvSeqlen, uint32_t gatheredStart, uint32_t &slot,
                                                  uint32_t &compactPos)
{
    if (compactPos > gatheredStart) {
        return false;
    }
    if (compactPos == gatheredStart) {
        return true;
    }
    uint32_t tokNeed = gatheredStart - compactPos;
    if (regionSize == STEP4_RS8 && (tokNeed % STEP4_RS8) == 0) {
        uint32_t liveNeed = tokNeed / STEP4_RS8;
        if (slot + liveNeed <= packedCount) {
            slot += liveNeed;
            compactPos = gatheredStart;
            return true;
        }
    }
    while (compactPos < gatheredStart) {
        uint32_t skipGm = 0;
        uint32_t st = PackedDecodeRs8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, kvSeqlen, slot,
                                                     compactPos, skipGm);
        (void)skipGm;
        if (st != 1) {
            return false;
        }
    }
    return compactPos == gatheredStart;
}

template <bool SUPPORT_PAIR, class PackedT>
__aicore__ inline void PackedRegionRs8ToUbListDecode(AscendC::LocalTensor<PackedT> ubPacked,
                                                                AscendC::LocalTensor<int32_t> ubGm, bool hasGm,
                                                                uint32_t packedCount, uint32_t regionSize,
                                                                uint32_t pageSize, uint32_t gatheredStart,
                                                                uint32_t gatheredLen, int32_t kvSeqlen,
                                                                uint32_t &compactPos, uint32_t &slot,
                                                                uint64_t tokenStride, uint32_t eleNumPerC0,
                                                                uint32_t ubOff, AscendC::LocalTensor<uint32_t> ubList,
                                                                uint32_t listBase, uint32_t relLimit)
{
    (void)ubGm;
    (void)hasGm;
    uint32_t nPair = 0;
    uint32_t nTail = 0;
    uint32_t rel = 1;
    uint32_t dealtLenAccum = 0;
    bool ok = PackedDecodeRs8SkipTo(ubPacked, packedCount, regionSize, pageSize, ubOff, kvSeqlen, gatheredStart,
                                         slot, compactPos);
    while (ok && slot < packedCount && dealtLenAccum < gatheredLen) {
        uint32_t gm0 = 0;
        uint32_t st0 = PackedDecodeRs8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, kvSeqlen,
                                                      slot, compactPos, gm0);
        if (st0 != 1) {
            ok = (st0 == 0 && dealtLenAccum == gatheredLen);
            break;
        }
        bool storedPair = false;
        if constexpr (SUPPORT_PAIR) {
            if (dealtLenAccum + 16u <= gatheredLen && rel + 3u <= relLimit) {
                uint32_t compactPos2 = compactPos;
                uint32_t slot2 = slot;
                uint32_t gm1 = 0;
                uint32_t st1 = PackedDecodeRs8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff,
                                                              kvSeqlen, slot2, compactPos2, gm1);
                if (st1 == 1 && gm1 != gm0) {
                    uint32_t gmLo = (gm0 < gm1) ? gm0 : gm1;
                    uint32_t gmHi = (gm0 < gm1) ? gm1 : gm0;
                    uint64_t stride = static_cast<uint64_t>(gmHi - gmLo) * tokenStride;
                    if (stride % eleNumPerC0 == 0) {
                        ubList.SetValue(listBase + rel, gmLo);
                        ubList.SetValue(listBase + rel + 1, static_cast<uint32_t>(stride));
                        ubList.SetValue(listBase + rel + 2, static_cast<uint32_t>(stride >> 32));
                        rel += 3;
                        compactPos = compactPos2;
                        slot = slot2;
                        dealtLenAccum += 16;
                        nPair += 1;
                        storedPair = true;
                    }
                }
            }
        }
        if (!storedPair) {
            if (dealtLenAccum + STEP4_RS8 < gatheredLen || rel + 1u > relLimit) {
                ok = false;
                break;
            }
            ubList.SetValue(listBase + rel, gm0);
            rel += 1;
            dealtLenAccum += STEP4_RS8;
            nTail += 1;
        }
    }
    uint32_t hdr = nPair | (nTail << STEP4_RS8_HDR_TAIL_SHIFT);
    ubList.SetValue(listBase, (ok && dealtLenAccum == gatheredLen) ? hdr : 0);
}

template <bool IS_DECODE, bool SUPPORT_PAIR, class PackedT>
__aicore__ inline void PackedRegionDecodeTileToUbList(AscendC::LocalTensor<PackedT> ubPacked,
                                                      AscendC::LocalTensor<int32_t> ubGm, bool hasGm,
                                                      uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                                      uint32_t gatheredStart, uint32_t gatheredLen, int32_t kvSeqlen,
                                                      uint32_t &compactPos, uint32_t &slot, uint64_t tokenStride,
                                                      uint32_t eleNumPerC0, uint32_t ubOff,
                                                      AscendC::LocalTensor<uint32_t> ubList, uint32_t listBase,
                                                      uint32_t relLimit)
{
    if (regionSize == STEP4_RS8) {
        if constexpr (IS_DECODE) {
            PackedRegionRs8ToUbListDecode<SUPPORT_PAIR>(
                ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize, gatheredStart, gatheredLen, kvSeqlen,
                compactPos, slot, tokenStride, eleNumPerC0, ubOff, ubList, listBase, relLimit);
        } else {
            PackedRegionRs8ToUbList<SUPPORT_PAIR>(
                ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize, gatheredStart, gatheredLen, compactPos, slot,
                tokenStride, eleNumPerC0, ubOff, ubList, listBase, relLimit);
        }
        return;
    }
    uint32_t nSeg = 0;
    uint32_t rel = 1;
    uint32_t dealtLenAccum = 0;
    uint32_t tileEnd = gatheredStart + gatheredLen;
    while (slot < packedCount && dealtLenAccum < gatheredLen && nSeg < STEP4_GATHER_MAX_SEG) {
        uint32_t oriStart = 0;
        uint32_t segLen = PackedNextSlotUb<IS_DECODE>(ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize,
                                                     gatheredStart, tileEnd, kvSeqlen, compactPos, slot, oriStart,
                                                     ubOff);
        if (segLen == 0) {
            break;
        }
        uint32_t ndNum = 1;
        uint32_t gmStart = oriStart;
        uint64_t srcStride = 0;
        if constexpr (SUPPORT_PAIR) {
            uint32_t compactPos2 = compactPos;
            uint32_t slot2 = slot;
            uint32_t oriStart2 = 0;
            uint32_t segLen2 = PackedNextSlotUb<IS_DECODE>(ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize,
                                                          gatheredStart, tileEnd, kvSeqlen, compactPos2, slot2,
                                                          oriStart2, ubOff);
            if (segLen2 == segLen && oriStart2 != oriStart) {
                uint32_t gmLo = (oriStart < oriStart2) ? oriStart : oriStart2;
                uint32_t gmHi = (oriStart < oriStart2) ? oriStart2 : oriStart;
                uint64_t stride = static_cast<uint64_t>(gmHi - gmLo) * tokenStride;
                if (stride % eleNumPerC0 == 0 && dealtLenAccum + segLen + segLen2 <= gatheredLen) {
                    ndNum = 2;
                    gmStart = gmLo;
                    srcStride = stride;
                    compactPos = compactPos2;
                    slot = slot2;
                    dealtLenAccum += segLen + segLen2;
                }
            }
        }
        if (ndNum == 1) {
            dealtLenAccum += segLen;
        }
        uint32_t next = PackedSegWriteRecordAt(ubList, listBase, rel, gmStart, segLen, ndNum, srcStride, relLimit);
        if (next == rel) {
            break;
        }
        rel = next;
        nSeg += 1;
    }
    ubList.SetValue(listBase, (dealtLenAccum < gatheredLen) ? 0 : nSeg);
}

template <class PackedT>
__aicore__ inline bool PackedRegionLoadPackedRowToUb(AscendC::GlobalTensor<PackedT> gPacked, uint32_t packedRow,
                                                    uint32_t totalTaskNum, uint32_t topK, uint32_t packedAlign,
                                                    uint32_t count, AscendC::LocalTensor<PackedT> ubPacked,
                                                    uint32_t &left)
{
    constexpr uint32_t ALIGN_ELEMS = static_cast<uint32_t>(32 / sizeof(PackedT));
    uint32_t totalPackedElems = totalTaskNum * topK;
    uint32_t packedRowStart = packedRow * topK;
    if (count == 0) {
        return false;
    }
    left = packedRowStart % ALIGN_ELEMS;
    uint32_t alignedStart = packedRowStart - left;
    if (alignedStart >= totalPackedElems) {
        return false;
    }
    uint32_t remain = totalPackedElems - alignedStart;
    uint32_t copyCount = (remain / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (copyCount > packedAlign) {
        copyCount = packedAlign;
    }
    if (copyCount < left + count) {
        return false;
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::DataCopy(ubPacked, gPacked[alignedStart], copyCount);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    return true;
}

// FA-loop decode stage: AIV0 writes half0, AIV1 writes half1. compactPos/slot persist
// across tiles so each AIV only skips the other half.
template <bool IS_DECODE, class PackedT>
__aicore__ inline void PackedRegionDecodeOneTileSplitToGm(AscendC::LocalTensor<PackedT> ubPacked,
                                                          AscendC::LocalTensor<int32_t> ubGm,
                                                          AscendC::LocalTensor<uint32_t> ubList, uint32_t count,
                                                          uint32_t regionSize, uint32_t pageSize, uint32_t kvBaseTile,
                                                          uint32_t gatheredTokens, uint32_t tileIdx, int32_t kvSeqlen,
                                                          uint32_t &compactPos, uint32_t &slot, uint64_t tokenStride,
                                                          uint32_t eleNumPerC0, uint32_t left, uint32_t l1Split,
                                                          __gm__ uint32_t *listPtr, uint32_t listWordOff)
{
    uint32_t start = tileIdx * kvBaseTile;
    uint32_t h = AscendC::GetSubBlockIdx();
    // Previous tile's DataCopy must have drained ubList before the scalar unit rewrites it.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    if (start >= gatheredTokens) {
        ubList.SetValue(0, 0);
        PackedRegionStoreUbListHalf(ubList, listPtr, listWordOff + h * STEP4_SEG_HALF_WORDS);
        return;
    }
    uint32_t len = kvBaseTile;
    if (start + len > gatheredTokens) {
        len = gatheredTokens - start;
    }
    uint32_t half = (l1Split > 0) ? l1Split : len;
    uint32_t nHalf = 1;
    if (half > 0 && len > half) {
        nHalf = 2;
    }
    if (h >= nHalf) {
        ubList.SetValue(0, 0);
        PackedRegionStoreUbListHalf(ubList, listPtr, listWordOff + h * STEP4_SEG_HALF_WORDS);
        return;
    }
    uint32_t halfStart = start + h * half;
    uint32_t halfLen = half;
    if (halfStart + halfLen > start + len) {
        halfLen = start + len - halfStart;
    }
    PackedRegionDecodeTileToUbList<IS_DECODE, true>(
        ubPacked, ubGm, false, count, regionSize, pageSize, halfStart, halfLen, kvSeqlen, compactPos, slot,
        tokenStride, eleNumPerC0, left, ubList, 0, STEP4_SEG_HALF_WORDS);
    PackedRegionStoreUbListHalf(ubList, listPtr, listWordOff + h * STEP4_SEG_HALF_WORDS);
}
template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline void PackedRegionIssuePairRs8(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                              uint32_t gm, uint64_t srcStride, uint32_t embed, uint32_t l1Dst,
                                              uint32_t dstStride)
{
    auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(l1Dst, 0), tla::MakeShape(STEP4_RS8, embed));
    auto gTile = tla::GetTile(gTensor, tla::MakeCoord(gm, 0), tla::MakeShape(STEP4_RS8, embed));
    copyGmToL1(l1Tile, gTile, 2, srcStride, dstStride);
}

template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline void PackedRegionIssueSingleRs8(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                uint32_t gm, uint32_t embed, uint32_t l1Dst)
{
    auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(l1Dst, 0), tla::MakeShape(STEP4_RS8, embed));
    auto gTile = tla::GetTile(gTensor, tla::MakeCoord(gm, 0), tla::MakeShape(STEP4_RS8, embed));
    copyGmToL1(l1Tile, gTile);
}

// rs=8 pair/tail list: hdr = nPair | (nTail << 16), then nPair * (gm, stride_lo, stride_hi), then nTail * gm.
// Snapshot all list GetValue into registers, then copyGmToL1 (PIPE_S vs MTE2). One MTE2 barrier at end.
__aicore__ inline bool PackedRegionRs8LoadHalf(AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                                    PackedRegionRs8Half &snap)
{
    uint32_t hdr = gList.GetValue(wordOff);
    if (hdr == 0) {
        snap.nPair = 0;
        snap.nTail = 0;
        return false;
    }
    uint32_t nPair = hdr & 0xFFu;
    uint32_t nTail = hdr >> STEP4_RS8_HDR_TAIL_SHIFT;
    if (nPair > STEP4_RS8_MAX_PAIR) {
        nPair = STEP4_RS8_MAX_PAIR;
    }
    if (nTail > STEP4_RS8_MAX_TAIL) {
        nTail = STEP4_RS8_MAX_TAIL;
    }
    snap.nPair = nPair;
    snap.nTail = nTail;
    uint32_t rel = 1;
    for (uint32_t i = 0; i < nPair; ++i) {
        snap.pairGm[i] = gList.GetValue(wordOff + rel);
        snap.pairLo[i] = gList.GetValue(wordOff + rel + 1);
        snap.pairHi[i] = gList.GetValue(wordOff + rel + 2);
        rel += 3;
    }
    for (uint32_t t = 0; t < nTail; ++t) {
        snap.tailGm[t] = gList.GetValue(wordOff + rel);
        rel += 1;
    }
    return true;
}

template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline void PackedRegionRs8IssueHalf(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                     PackedRegionRs8Half const &snap, uint32_t embed,
                                                     uint32_t l1DstBase)
{
    const uint32_t dstStride = STEP4_RS8 * tla::get<0, 0>(l1Tensor.stride());
    constexpr uint32_t kPairRows = 2 * STEP4_RS8;
    uint32_t localDst = l1DstBase;
    for (uint32_t i = 0; i < snap.nPair; ++i) {
        uint64_t srcStride = (static_cast<uint64_t>(snap.pairHi[i]) << 32) | static_cast<uint64_t>(snap.pairLo[i]);
        PackedRegionIssuePairRs8(copyGmToL1, gTensor, l1Tensor, snap.pairGm[i], srcStride, embed, localDst, dstStride);
        localDst += kPairRows;
    }
    for (uint32_t t = 0; t < snap.nTail; ++t) {
        PackedRegionIssueSingleRs8(copyGmToL1, gTensor, l1Tensor, snap.tailGm[t], embed, localDst);
        localDst += STEP4_RS8;
    }
    if (snap.nPair + snap.nTail > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
}

template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline void PackedRegionIssueFromGmRs8(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                       AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                                       uint32_t embed, uint32_t l1DstBase)
{
    PackedRegionRs8Half snap;
    if (!PackedRegionRs8LoadHalf(gList, wordOff, snap)) {
        return;
    }
    PackedRegionRs8IssueHalf(copyGmToL1, gTensor, l1Tensor, snap, embed, l1DstBase);
}
template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromGmGeneric(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                          AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                                          uint32_t embed, uint32_t l1DstBase)
{
    uint32_t n = gList.GetValue(wordOff);
    if (n == 0 || n > STEP4_GATHER_MAX_SEG) {
        return 0;
    }
    const uint32_t eleNumPerC0 = tla::get<0, 0>(l1Tensor.stride());
    uint32_t rel = 1;
    uint32_t localDst = 0;
    uint32_t issued = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (rel + 2u > STEP4_SEG_HALF_WORDS) {
            break;
        }
        uint32_t gm = gList.GetValue(wordOff + rel);
        uint32_t meta = gList.GetValue(wordOff + rel + 1);
        rel += 2;
        uint32_t nValue = 0;
        uint32_t ndNum = 0;
        PackedSegUnpackMeta(meta, nValue, ndNum);
        uint64_t srcStride = 0;
        if (ndNum == 2) {
            if (rel + 2u > STEP4_SEG_HALF_WORDS) {
                break;
            }
            uint32_t lo = gList.GetValue(wordOff + rel);
            uint32_t hi = gList.GetValue(wordOff + rel + 1);
            rel += 2;
            srcStride = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
        } else {
            ndNum = 1;
        }
        auto l1Tile =
            tla::GetTile(l1Tensor, tla::MakeCoord(l1DstBase + localDst, 0), tla::MakeShape(nValue, embed));
        auto gTile = tla::GetTile(gTensor, tla::MakeCoord(gm, 0), tla::MakeShape(nValue, embed));
        if (ndNum == 2) {
            copyGmToL1(l1Tile, gTile, 2, srcStride, nValue * eleNumPerC0);
        } else {
            copyGmToL1(l1Tile, gTile);
        }
        localDst += nValue * ndNum;
        issued += nValue * ndNum;
    }
    if (issued > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    return issued;
}

// Same generic list as K; each ndNum=2 becomes two nValue copies. V 256-row zN NaNs on pair DMA.
template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromGmGenericSingles(CopyGmToL1 &copyGmToL1, TensorGM &gTensor,
                                                                 TensorL1 &l1Tensor,
                                                                 AscendC::GlobalTensor<uint32_t> gList,
                                                                 uint32_t wordOff, uint32_t embed, uint32_t l1DstBase)
{
    uint32_t n = gList.GetValue(wordOff);
    if (n == 0 || n > STEP4_GATHER_MAX_SEG) {
        return 0;
    }
    uint64_t tokenStride = static_cast<uint64_t>(tla::get<0>(gTensor.stride()));
    uint32_t rel = 1;
    uint32_t localDst = 0;
    uint32_t issued = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (rel + 2u > STEP4_SEG_HALF_WORDS) {
            break;
        }
        uint32_t gm = gList.GetValue(wordOff + rel);
        uint32_t meta = gList.GetValue(wordOff + rel + 1);
        rel += 2;
        uint32_t nValue = 0;
        uint32_t ndNum = 0;
        PackedSegUnpackMeta(meta, nValue, ndNum);
        uint64_t srcStride = 0;
        if (ndNum == 2) {
            if (rel + 2u > STEP4_SEG_HALF_WORDS) {
                break;
            }
            uint32_t lo = gList.GetValue(wordOff + rel);
            uint32_t hi = gList.GetValue(wordOff + rel + 1);
            rel += 2;
            srcStride = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
        } else {
            ndNum = 1;
        }
        auto l1Tile =
            tla::GetTile(l1Tensor, tla::MakeCoord(l1DstBase + localDst, 0), tla::MakeShape(nValue, embed));
        auto gTile = tla::GetTile(gTensor, tla::MakeCoord(gm, 0), tla::MakeShape(nValue, embed));
        copyGmToL1(l1Tile, gTile);
        localDst += nValue;
        issued += nValue;
        if (ndNum == 2) {
            uint32_t gmHi = gm;
            if (tokenStride > 0) {
                gmHi = gm + static_cast<uint32_t>(srcStride / tokenStride);
            }
            auto l1Tile2 =
                tla::GetTile(l1Tensor, tla::MakeCoord(l1DstBase + localDst, 0), tla::MakeShape(nValue, embed));
            auto gTile2 = tla::GetTile(gTensor, tla::MakeCoord(gmHi, 0), tla::MakeShape(nValue, embed));
            copyGmToL1(l1Tile2, gTile2);
            localDst += nValue;
            issued += nValue;
        }
    }
    if (issued > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    return issued;
}

template <bool IS_DECODE, class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromGm(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                   AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                                   uint32_t embed, uint32_t l1DstBase, uint32_t regionSize)
{
    (void)IS_DECODE;
    // Prefill and Decode rs=8 share the same rs=8 pair/tail list format on GM.
    if (regionSize == STEP4_RS8) {
        PackedRegionIssueFromGmRs8(copyGmToL1, gTensor, l1Tensor, gList, wordOff, embed, l1DstBase);
        return 0;
    }
    return PackedRegionIssueFromGmGeneric(copyGmToL1, gTensor, l1Tensor, gList, wordOff, embed, l1DstBase);
}
// Compact live tokens from packed slots into a contiguous L1 tile.
// Prefill INT32: phys_region | (valid_tokens << 24); skip sentinel is -1 (0xFFFFFFFF).
// Decode INT64: start_token | (phys_region << 32); valid = clamp(kv_seqlen - start, 0, regionSize).
// One slot per segment; two equal-length slots use Nd2Nz ndNum=2. compactPos/slot are inout
// cursors. Issue copies without a per-segment PipeBarrier; wait once after the tile.
template <bool IS_DECODE, class CopyGmToL1, class TensorGM, class TensorL1, class PackedT>
__aicore__ inline void PackedRegionGatherToL1(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                              AscendC::GlobalTensor<PackedT> gPacked, uint32_t packedRowStart,
                                              uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                              uint32_t gatheredStart, uint32_t gatheredLen, uint32_t embed,
                                              int32_t kvSeqlen, uint32_t &compactPos, uint32_t &slot)
{
    constexpr bool kSupportPair = tla::detail::isRowMajor<typename TensorGM::Layout>::value &&
                                  !AscendC::IsSameType<typename TensorGM::Element, fp4x2_e2m1_t>::value;
    uint32_t targetEnd = gatheredStart + gatheredLen;
    uint32_t dealtLenAccum = 0;
    while (slot < packedCount && dealtLenAccum < gatheredLen) {
        uint32_t oriStart = 0;
        uint32_t segLen = PackedNextSlot<IS_DECODE>(gPacked, packedRowStart, packedCount, regionSize, pageSize,
                                                    gatheredStart, targetEnd, kvSeqlen, compactPos, slot, oriStart);
        if (segLen == 0) {
            break;
        }
        auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(segLen, embed));
        auto gTile = tla::GetTile(gTensor, tla::MakeCoord(oriStart, 0), tla::MakeShape(segLen, embed));
        bool paired = false;
        if constexpr (kSupportPair) {
            uint32_t compactPos2 = compactPos;
            uint32_t slot2 = slot;
            uint32_t oriStart2 = 0;
            uint32_t segLen2 = PackedNextSlot<IS_DECODE>(gPacked, packedRowStart, packedCount, regionSize, pageSize,
                                                         gatheredStart, targetEnd, kvSeqlen, compactPos2, slot2,
                                                         oriStart2);
            if (segLen2 == segLen && oriStart2 != oriStart) {
                uint32_t gmLo = (oriStart < oriStart2) ? oriStart : oriStart2;
                uint32_t gmHi = (oriStart < oriStart2) ? oriStart2 : oriStart;
                uint64_t srcStride = static_cast<uint64_t>(gmHi - gmLo) *
                                     static_cast<uint64_t>(tla::get<0>(gTensor.stride()));
                const uint32_t eleNumPerC0 = tla::get<0, 0>(l1Tensor.stride());
                const uint32_t dstStride = segLen * eleNumPerC0;
                if (srcStride % eleNumPerC0 == 0 && dealtLenAccum + segLen + segLen2 <= gatheredLen) {
                    auto gPair = tla::GetTile(gTensor, tla::MakeCoord(gmLo, 0), tla::MakeShape(segLen, embed));
                    copyGmToL1(l1Tile, gPair, 2, srcStride, dstStride);
                    dealtLenAccum += segLen + segLen2;
                    compactPos = compactPos2;
                    slot = slot2;
                    paired = true;
                }
            }
        }
        if (!paired) {
            copyGmToL1(l1Tile, gTile);
            dealtLenAccum += segLen;
        }
    }
    if (dealtLenAccum > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
}

} // namespace NpuArch::Gemm::Block

#endif
