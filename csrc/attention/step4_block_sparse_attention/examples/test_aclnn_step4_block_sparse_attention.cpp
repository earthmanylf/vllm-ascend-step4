/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Prefill / Decode BF16 & FP16 accuracy check vs FP32 CPU golden (same packing contract as Triton ref).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_step4_block_sparse_attention.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)

namespace {

constexpr uint32_t REGION_ID_MASK = 0x00FFFFFFu;
constexpr uint32_t VALID_SHIFT = 24;
constexpr uint32_t PREFILL_SKIP = 0xFFFFFFFFu;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto d : shape) {
        size *= d;
    }
    return size;
}

uint16_t FloatToBf16(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t roundBit = (bits >> 15) & 1u;
    uint32_t sticky = bits & 0x7FFFu;
    uint32_t rounding = (sticky != 0u) ? 1u : roundBit;
    return static_cast<uint16_t>((bits >> 16) + rounding);
}

float Bf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// FP16 <-> FP32 bit-level conversion. Round-to-nearest-even; sufficient for the
// |x| <= 1 test data range (fp16-normal, no subnormal inputs).
uint16_t FloatToFp16(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t abs = bits & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) { // NaN
        return static_cast<uint16_t>(sign | 0x7E00u);
    }
    if (abs >= 0x477FF000u) { // overflow -> Inf
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (abs < 0x33000000u) { // underflow -> 0
        return static_cast<uint16_t>(sign);
    }
    if (abs < 0x38800000u) { // fp16 subnormal: value = sub * 2^-24
        uint32_t m24 = (abs & 0x7FFFFFu) | 0x800000u;       // 24-bit significand
        int32_t e = static_cast<int32_t>(abs >> 23) - 127;   // unbiased exponent, [-25, -15]
        uint32_t shift = static_cast<uint32_t>(-e - 1);      // [14, 24]
        uint32_t sub = m24 >> shift;
        uint32_t rem = m24 & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (sub & 1u) != 0u)) {
            sub += 1u;
        }
        return static_cast<uint16_t>(sign | sub);
    }
    uint32_t exp = (abs >> 23) - 127u + 15u;
    uint32_t frac = abs & 0x7FFFFFu;
    uint32_t val = (exp << 10) | (frac >> 13);
    uint32_t rem = frac & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (val & 1u) != 0u)) {
        val += 1u; // carry into exponent is the correct rounding
    }
    return static_cast<uint16_t>(sign | val);
}

float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign; // ±0
        } else { // subnormal -> normalized fp32
            int e = -1;
            uint32_t f = frac;
            while ((f & 0x400u) == 0) {
                f <<= 1;
                --e;
            }
            f &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(127 - 15 + e) << 23) | (f << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (frac << 13); // Inf / NaN
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (frac << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, static_cast<size_t>(size), hostData.data(), static_cast<size_t>(size),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void FillBf16(std::vector<uint16_t> &dst, float scale, float offset, uint16_t (*fromFloat)(float))
{
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] = fromFloat(std::sin(static_cast<float>(i) * scale + offset) * 0.25f);
    }
}

void FillBf16Randn(std::vector<uint16_t> &dst, uint32_t seed, uint16_t (*fromFloat)(float))
{
    uint32_t rng = seed;
    for (size_t i = 0; i < dst.size(); ++i) {
        rng = rng * 1664525u + 1013904223u;
        float u = static_cast<float>(rng >> 8) * (1.0f / 16777216.0f);
        dst[i] = fromFloat(u * 2.0f - 1.0f);
    }
}

void FillBf16Const(std::vector<uint16_t> &dst, float value, uint16_t (*fromFloat)(float))
{
    uint16_t h = fromFloat(value);
    std::fill(dst.begin(), dst.end(), h);
}

constexpr uint64_t DECODE_TAIL_SENTINEL = 0x7FFFFFFFFFFFFFFFULL;

inline int32_t EncodePrefillSlot(int64_t phys, uint32_t valid)
{
    uint32_t word = (static_cast<uint32_t>(phys) & REGION_ID_MASK) | (valid << VALID_SHIFT);
    int32_t bits = 0;
    std::memcpy(&bits, &word, sizeof(bits));
    return bits;
}

void GatherTokensPrefill(const std::vector<int32_t> &packed, const std::vector<int32_t> &counts, int64_t row,
                         int64_t topK, int64_t regionSize, std::vector<int64_t> &tokens)
{
    tokens.clear();
    int32_t count = counts[static_cast<size_t>(row)];
    for (int32_t slot = 0; slot < count; ++slot) {
        uint32_t word = static_cast<uint32_t>(packed[static_cast<size_t>(row * topK + slot)]);
        if (word == PREFILL_SKIP) {
            continue;
        }
        uint32_t phys = word & REGION_ID_MASK;
        uint32_t valid = word >> VALID_SHIFT;
        if (valid == 0) {
            continue;
        }
        for (uint32_t j = 0; j < valid; ++j) {
            tokens.push_back(static_cast<int64_t>(phys) * regionSize + static_cast<int64_t>(j));
        }
    }
}

void GatherTokensDecode(const std::vector<int64_t> &packed, const std::vector<int32_t> &counts, int64_t row,
                        int64_t topK, int64_t regionSize, int32_t kvSeqlen, std::vector<int64_t> &tokens)
{
    tokens.clear();
    int32_t count = counts[static_cast<size_t>(row)];
    for (int32_t slot = 0; slot < count; ++slot) {
        uint64_t word = static_cast<uint64_t>(packed[static_cast<size_t>(row * topK + slot)]);
        uint32_t start = static_cast<uint32_t>(word);
        uint32_t phys = static_cast<uint32_t>(word >> 32);
        int64_t vis = static_cast<int64_t>(kvSeqlen) - static_cast<int64_t>(start);
        if (vis < 0) {
            vis = 0;
        }
        if (vis > regionSize) {
            vis = regionSize;
        }
        for (int64_t j = 0; j < vis; ++j) {
            tokens.push_back(static_cast<int64_t>(phys) * regionSize + j);
        }
    }
}

void CpuGolden(const std::vector<uint16_t> &qBf, const std::vector<uint16_t> &kBf, const std::vector<uint16_t> &vBf,
               const std::vector<std::vector<int64_t>> &rowTokens, int64_t qT, int64_t kvT, int64_t n1, int64_t g,
               int64_t d, float scale, float (*toFloat)(uint16_t), std::vector<float> &out,
               std::vector<float> &lse)
{
    int64_t groupSize = n1 / g;
    out.assign(static_cast<size_t>(qT * n1 * d), 0.0f);
    lse.assign(static_cast<size_t>(qT * n1), 0.0f);
    std::vector<float> scores;
    for (int64_t kvGroup = 0; kvGroup < g; ++kvGroup) {
        for (int64_t qIdx = 0; qIdx < qT; ++qIdx) {
            int64_t row = kvGroup * qT + qIdx;
            const std::vector<int64_t> &tokens = rowTokens[static_cast<size_t>(row)];
            int64_t s = static_cast<int64_t>(tokens.size());
            if (s == 0) {
                continue;
            }
            scores.assign(static_cast<size_t>(groupSize * s), 0.0f);
            for (int64_t h = 0; h < groupSize; ++h) {
                int64_t qOff = ((qIdx * n1) + kvGroup * groupSize + h) * d;
                float maxv = -INFINITY;
                for (int64_t t = 0; t < s; ++t) {
                    int64_t kOff = (tokens[static_cast<size_t>(t)] * g + kvGroup) * d;
                    float acc = 0.0f;
                    for (int64_t c = 0; c < d; ++c) {
                        acc += toFloat(qBf[static_cast<size_t>(qOff + c)]) *
                               toFloat(kBf[static_cast<size_t>(kOff + c)]);
                    }
                    acc *= scale;
                    scores[static_cast<size_t>(h * s + t)] = acc;
                    maxv = std::max(maxv, acc);
                }
                float sum = 0.0f;
                for (int64_t t = 0; t < s; ++t) {
                    float e = std::exp(scores[static_cast<size_t>(h * s + t)] - maxv);
                    scores[static_cast<size_t>(h * s + t)] = e;
                    sum += e;
                }
                lse[static_cast<size_t>(qIdx * n1 + kvGroup * groupSize + h)] = maxv + std::log(sum);
                for (int64_t t = 0; t < s; ++t) {
                    float p = scores[static_cast<size_t>(h * s + t)] / sum;
                    int64_t vOff = (tokens[static_cast<size_t>(t)] * g + kvGroup) * d;
                    for (int64_t c = 0; c < d; ++c) {
                        out[static_cast<size_t>(qOff + c)] += p * toFloat(vBf[static_cast<size_t>(vOff + c)]);
                    }
                }
            }
        }
    }
    (void)kvT;
}

bool Compare(const char *name, const std::vector<float> &golden, const std::vector<float> &got, float rtol, float atol)
{
    double maxAbs = 0.0;
    double maxRel = 0.0;
    int64_t fail = 0;
    int64_t n = static_cast<int64_t>(golden.size());
    for (int64_t i = 0; i < n; ++i) {
        float a = golden[static_cast<size_t>(i)];
        float b = got[static_cast<size_t>(i)];
        float diff = std::fabs(a - b);
        float denom = std::max(std::fabs(a), std::fabs(b));
        float rel = (denom > 1e-12f) ? diff / denom : diff;
        maxAbs = std::max(maxAbs, static_cast<double>(diff));
        maxRel = std::max(maxRel, static_cast<double>(rel));
        if (diff > atol + rtol * std::fabs(a)) {
            ++fail;
            if (fail <= 5) {
                LOG_PRINT("  %s mismatch[%ld] golden=%g got=%g abs=%g rel=%g\n", name, i, a, b, diff, rel);
            }
        }
    }
    LOG_PRINT("%s: n=%ld fail=%ld max_abs=%.6g max_rel=%.6g (rtol=%g atol=%g) %s\n", name, n, fail, maxAbs, maxRel,
              rtol, atol, fail == 0 ? "PASS" : "FAIL");
    return fail == 0;
}

struct Case {
    int64_t qT;
    int64_t kvT;
    int64_t n1;
    int64_t g;
    int64_t d;
    int64_t regionSize;
    int64_t topK;
    bool partialTail;
    bool emptyFirst;
    int64_t nLiveRegions; // 0 -> default 3
    int64_t sparseMode;   // 0 prefill, 1 decode
    int64_t kvSeqlen;     // decode; 0 -> kvT
    int64_t skipStride;   // 1 consecutive phys, 2 every other
    bool varySeqlen;      // decode: seqlens[q] decreases with q
    int fillMode;         // 0 sine, 1 randn, 2 constant-K (O = mean V)
    bool reverseSlots;    // softmax is permutation-invariant
    bool inlineSentinels; // decode: -1 / 0x7FFF... inside region_counts
    bool useFp16;         // false (default) -> BF16 QKV, true -> FP16 QKV
};

int RunCase(aclrtStream stream, const Case &cs)
{
    int64_t qT = cs.qT;
    int64_t kvT = cs.kvT;
    int64_t n1 = cs.n1;
    int64_t g = cs.g;
    int64_t d = cs.d;
    int64_t regionSize = cs.regionSize;
    int64_t topK = cs.topK;
    int64_t sparseMode = cs.sparseMode;
    const bool isDecode = (sparseMode == 1);
    int32_t kvSeqlen = static_cast<int32_t>(kvT);
    if (isDecode) {
        kvSeqlen = (cs.kvSeqlen >= 0) ? static_cast<int32_t>(cs.kvSeqlen) : static_cast<int32_t>(kvT);
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const bool useFp16 = cs.useFp16;
    uint16_t (*fromFloat)(float) = useFp16 ? &FloatToFp16 : &FloatToBf16;
    float (*toFloat)(uint16_t) = useFp16 ? &Fp16ToFloat : &Bf16ToFloat;
    const aclDataType qkvType = useFp16 ? ACL_FLOAT16 : ACL_BF16;

    std::vector<int64_t> qShape = {qT, n1, d};
    std::vector<int64_t> kvShape = {kvT, g, d};
    std::vector<int64_t> packedShape = {g * qT, topK};
    std::vector<int64_t> countShape = {g * qT};
    std::vector<int64_t> seqShape = {qT};
    std::vector<int64_t> lseShape = {qT, n1};

    std::vector<uint16_t> qHost(static_cast<size_t>(GetShapeSize(qShape)));
    std::vector<uint16_t> kHost(static_cast<size_t>(GetShapeSize(kvShape)));
    std::vector<uint16_t> vHost(static_cast<size_t>(GetShapeSize(kvShape)));
    if (cs.fillMode == 1) {
        FillBf16Randn(qHost, 1, fromFloat);
        FillBf16Randn(kHost, 2, fromFloat);
        FillBf16Randn(vHost, 3, fromFloat);
    } else if (cs.fillMode == 2) {
        FillBf16(qHost, 0.017f, 0.1f, fromFloat);
        FillBf16Const(kHost, 1.0f, fromFloat);
        FillBf16(vHost, 0.011f, 0.7f, fromFloat);
    } else {
        FillBf16(qHost, 0.017f, 0.1f, fromFloat);
        FillBf16(kHost, 0.013f, 0.3f, fromFloat);
        FillBf16(vHost, 0.011f, 0.7f, fromFloat);
    }

    std::vector<int32_t> packedI32(static_cast<size_t>(g * qT * topK), -1);
    std::vector<int64_t> packedI64(static_cast<size_t>(g * qT * topK), 0);
    std::vector<int32_t> counts(static_cast<size_t>(g * qT), 0);
    std::vector<int32_t> seqlens(static_cast<size_t>(qT), kvSeqlen);
    if (cs.varySeqlen) {
        for (int64_t qIdx = 0; qIdx < qT; ++qIdx) {
            int32_t s = kvSeqlen - static_cast<int32_t>(qIdx) * 48;
            seqlens[static_cast<size_t>(qIdx)] = std::max(1, s);
        }
    }
    int64_t nRegion = kvT / regionSize;
    int64_t stride = cs.skipStride > 1 ? cs.skipStride : 1;
    for (int64_t kvGroup = 0; kvGroup < g; ++kvGroup) {
        for (int64_t qIdx = 0; qIdx < qT; ++qIdx) {
            int64_t row = kvGroup * qT + qIdx;
            if (cs.emptyFirst && qIdx == 0) {
                counts[static_cast<size_t>(row)] = 0;
                continue;
            }
            int32_t nSlot = 0;
            int64_t defaultLive = 3;
            int64_t live = cs.nLiveRegions > 0 ? cs.nLiveRegions : defaultLive;
            live = std::min(live, std::min(topK, (nRegion + stride - 1) / stride));
            for (int64_t i = 0; i < live; ++i) {
                int64_t r = i * stride;
                if (r >= nRegion) {
                    break;
                }
                if (isDecode) {
                    uint32_t start = static_cast<uint32_t>(r * regionSize);
                    uint32_t phys = static_cast<uint32_t>(r);
                    packedI64[static_cast<size_t>(row * topK + nSlot)] =
                        static_cast<int64_t>(static_cast<uint64_t>(start) | (static_cast<uint64_t>(phys) << 32));
                } else {
                    uint32_t valid = static_cast<uint32_t>(regionSize);
                    if (cs.partialTail && i == live - 1) {
                        valid = 3;
                    }
                    packedI32[static_cast<size_t>(row * topK + nSlot)] = EncodePrefillSlot(r, valid);
                }
                ++nSlot;
            }
            if (cs.inlineSentinels && isDecode && nSlot + 2 <= topK) {
                packedI64[static_cast<size_t>(row * topK + nSlot)] = -1;
                ++nSlot;
                packedI64[static_cast<size_t>(row * topK + nSlot)] =
                    static_cast<int64_t>(DECODE_TAIL_SENTINEL);
                ++nSlot;
            }
            if (cs.reverseSlots && nSlot > 1) {
                if (isDecode) {
                    std::reverse(packedI64.begin() + row * topK, packedI64.begin() + row * topK + nSlot);
                } else {
                    std::reverse(packedI32.begin() + row * topK, packedI32.begin() + row * topK + nSlot);
                }
            }
            counts[static_cast<size_t>(row)] = nSlot;
        }
    }

    std::vector<std::vector<int64_t>> rowTokens(static_cast<size_t>(g * qT));
    for (int64_t row = 0; row < g * qT; ++row) {
        int32_t rowSeqlen = seqlens[static_cast<size_t>(row % qT)];
        if (isDecode) {
            GatherTokensDecode(packedI64, counts, row, topK, regionSize, rowSeqlen, rowTokens[static_cast<size_t>(row)]);
        } else {
            GatherTokensPrefill(packedI32, counts, row, topK, regionSize, rowTokens[static_cast<size_t>(row)]);
        }
    }

    LOG_PRINT("cpu golden start (dtype=%s mode=%ld qT=%ld G=%ld N1=%ld D=%ld rs=%ld topK=%ld kvT=%ld kvSeqlen=%d "
              "stride=%ld)\n",
              useFp16 ? "FP16" : "BF16", sparseMode, qT, g, n1, d, regionSize, topK, kvT, kvSeqlen, stride);
    std::vector<float> goldenO;
    std::vector<float> goldenLse;
    CpuGolden(qHost, kHost, vHost, rowTokens, qT, kvT, n1, g, d, scale, toFloat, goldenO, goldenLse);
    LOG_PRINT("cpu golden done, launching kernel\n");

    void *qDev = nullptr;
    void *kDev = nullptr;
    void *vDev = nullptr;
    void *packedDev = nullptr;
    void *countDev = nullptr;
    void *seqDev = nullptr;
    void *oDev = nullptr;
    void *lseDev = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *packedT = nullptr;
    aclTensor *countT = nullptr;
    aclTensor *seqT = nullptr;
    aclTensor *o = nullptr;
    aclTensor *lse = nullptr;

    int ret = CreateAclTensor(qHost, qShape, &qDev, qkvType, &q);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(kHost, kvShape, &kDev, qkvType, &k);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(vHost, kvShape, &vDev, qkvType, &v);
    CHECK_RET(ret == 0, return ret);
    if (isDecode) {
        ret = CreateAclTensor(packedI64, packedShape, &packedDev, ACL_INT64, &packedT);
        CHECK_RET(ret == 0, return ret);
        ret = CreateAclTensor(seqlens, seqShape, &seqDev, ACL_INT32, &seqT);
        CHECK_RET(ret == 0, return ret);
    } else {
        ret = CreateAclTensor(packedI32, packedShape, &packedDev, ACL_INT32, &packedT);
        CHECK_RET(ret == 0, return ret);
    }
    ret = CreateAclTensor(counts, countShape, &countDev, ACL_INT32, &countT);
    CHECK_RET(ret == 0, return ret);

    std::vector<uint16_t> oHost(static_cast<size_t>(GetShapeSize(qShape)), 0);
    std::vector<float> lseHost(static_cast<size_t>(GetShapeSize(lseShape)), 0.0f);
    ret = CreateAclTensor(oHost, qShape, &oDev, qkvType, &o);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(lseHost, lseShape, &lseDev, ACL_FLOAT, &lse);
    CHECK_RET(ret == 0, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    char layoutQ[] = "TND";
    char layoutKv[] = "TGD";
    auto aclRet = aclnnStep4BlockSparseAttentionGetWorkspaceSize(
        q, k, v, packedT, countT, seqT, n1, g, regionSize, 128, static_cast<double>(scale), layoutQ, layoutKv, 1, 1, o,
        lse, &workspaceSize, &executor);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", aclRet); return aclRet);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclRet = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("malloc workspace failed. ERROR: %d\n", aclRet); return aclRet);
    }
    LOG_PRINT("kernel launch, workspace=%lu\n", static_cast<unsigned long>(workspaceSize));
    aclRet = aclnnStep4BlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclnnStep4BlockSparseAttention failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("kernel launched, sync stream (timeout 180s)\n");
    aclRet = aclrtSynchronizeStreamWithTimeout(stream, 180000);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("synchronize failed. ERROR: %d\n", aclRet); return aclRet);
    LOG_PRINT("kernel done\n");

    aclRet = aclrtMemcpy(oHost.data(), oHost.size() * sizeof(uint16_t), oDev, oHost.size() * sizeof(uint16_t),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy O failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(lseHost.data(), lseHost.size() * sizeof(float), lseDev, lseHost.size() * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy LSE failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> oFp(oHost.size());
    for (size_t i = 0; i < oHost.size(); ++i) {
        oFp[i] = toFloat(oHost[i]);
    }

    bool ok = Compare("attention_out", goldenO, oFp, 1e-2f, 4e-3f);
    ok = Compare("softmax_lse", goldenLse, lseHost, 1e-3f, 1e-3f) && ok;

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(packedT);
    aclDestroyTensor(countT);
    if (seqT != nullptr) {
        aclDestroyTensor(seqT);
    }
    aclDestroyTensor(o);
    aclDestroyTensor(lse);
    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(vDev);
    aclrtFree(packedDev);
    aclrtFree(countDev);
    if (seqDev != nullptr) {
        aclrtFree(seqDev);
    }
    aclrtFree(oDev);
    aclrtFree(lseDev);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return ok ? 0 : 1;
}

} // namespace

int main()
{
    int32_t deviceId = 2;
    const char *envId = std::getenv("TEST_DEVICE_ID");
    if (envId != nullptr) {
        deviceId = std::atoi(envId);
    }
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<Case> cases = {
        // Prefill: 1 / 2 KV tiles, partial tail, empty row, GQA, skipped phys, model-scale
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 512, 16, 1, 192, 8, 64, false, false, 64, 0, 0, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, true, false, 32, 0, 0, 1, false, 0, false, false},
        {2, 256, 16, 1, 192, 8, 32, false, true, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 64, 4, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 16, 0, 0, 2, false, 0, false, false},
        {1, 4104, 64, 4, 192, 8, 513, false, false, 513, 0, 0, 1, false, 0, false, false},
        // Prefill: qT across qBaseTile=16, single region, empty-all, unaligned kvT, GQA 32/2, D=128, regionSize=16
        {8, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {16, 256, 16, 1, 192, 8, 16, false, false, 16, 0, 0, 1, false, 0, false, false},
        {17, 256, 16, 1, 192, 8, 16, false, false, 16, 0, 0, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 1, 0, 0, 1, false, 0, false, false},
        {1, 8, 16, 1, 192, 8, 4, false, false, 1, 0, 0, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, true, 32, 0, 0, 1, false, 0, false, false},
        {1, 264, 16, 1, 192, 8, 64, true, false, 33, 0, 0, 1, false, 0, false, false},
        {1, 256, 32, 2, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 16, 1, 128, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 16, 16, false, false, 16, 0, 0, 1, false, 0, false, false},
        // Decode: tiles, partial via kv_seqlen, GQA, batch, empty, skip, vary seqlens
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {1, 512, 16, 1, 192, 8, 64, false, false, 64, 1, 512, 1, false, 0, false, false},
        {1, 264, 16, 1, 192, 8, 64, false, false, 33, 1, 259, 1, false, 0, false, false},
        {1, 256, 64, 4, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {4, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {2, 256, 16, 1, 192, 8, 32, false, true, 32, 1, 256, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 16, 1, 256, 2, false, 0, false, false},
        {4, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, true, 0, false, false},
        {1, 512, 64, 4, 192, 8, 64, false, false, 64, 1, 512, 1, false, 0, false, false},
        // Decode: larger batch, kv_seqlen edges, nLive=1, GQA 32/2, D=128, regionSize=16
        {8, 256, 16, 1, 192, 8, 16, false, false, 16, 1, 256, 1, false, 0, false, false},
        {16, 256, 16, 1, 192, 8, 16, false, false, 16, 1, 256, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 0, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 1, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 8, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 9, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 1, 1, 256, 1, false, 0, false, false},
        {1, 256, 32, 2, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {1, 256, 16, 1, 128, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {1, 256, 16, 1, 192, 16, 16, false, false, 16, 1, 256, 1, false, 0, false, false},
        {8, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, true, 0, false, false},
        // Independent checks: randn, constant-K identity, slot reverse, decode sentinels
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 1, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 1, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 2, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 2, false, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, true, false},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, true, false},
        {1, 256, 16, 1, 192, 8, 64, false, false, 32, 1, 256, 1, false, 0, false, true},
        // groupSize != qBaseTile(16): 1 / 4 / 8 / 32, prefill + decode
        {1, 256, 1, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 4, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 8, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 16, 2, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 32, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false},
        {1, 256, 1, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {1, 256, 4, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {1, 256, 8, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {1, 256, 16, 2, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        {1, 256, 32, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false},
        // FP16 Prefill: tiles, partial tail, GQA, skip, empty-all, unaligned kvT, D=128, rs=16, randn/const-K/rev
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 512, 16, 1, 192, 8, 64, false, false, 64, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, true, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 64, 4, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 16, 0, 0, 2, false, 0, false, false, true},
        {1, 264, 16, 1, 192, 8, 64, true, false, 33, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 32, 2, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 128, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 16, 16, false, false, 16, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 1, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 2, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, true, false, true},
        // FP16 Decode: tiles, partial, GQA, batch, empty, seqlen edges, sentinels, randn
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false, true},
        {1, 512, 16, 1, 192, 8, 64, false, false, 64, 1, 512, 1, false, 0, false, false, true},
        {1, 264, 16, 1, 192, 8, 64, false, false, 33, 1, 259, 1, false, 0, false, false, true},
        {1, 256, 64, 4, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false, true},
        {4, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false, true},
        {2, 256, 16, 1, 192, 8, 32, false, true, 32, 1, 256, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 0, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 1, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 8, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 9, 1, false, 0, false, false, true},
        {1, 256, 16, 1, 192, 8, 64, false, false, 32, 1, 256, 1, false, 0, false, true, true},
        {1, 256, 16, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 1, false, false, true},
        // FP16 groupSize != qBaseTile(16): 1 / 4 / 8 / 32, prefill + decode
        {1, 256, 1, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 4, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 8, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 32, 1, 192, 8, 32, false, false, 32, 0, 0, 1, false, 0, false, false, true},
        {1, 256, 1, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false, true},
        {1, 256, 4, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false, true},
        {1, 256, 8, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false, true},
        {1, 256, 32, 1, 192, 8, 32, false, false, 32, 1, 256, 1, false, 0, false, false, true},
    };
    int failed = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        LOG_PRINT("=== case %zu dtype=%s mode=%ld qT=%ld kvT=%ld n1=%ld g=%ld d=%ld rs=%ld live=%ld kvSeqlen=%ld "
                  "fill=%d rev=%d sent=%d ===\n",
                  i, cases[i].useFp16 ? "FP16" : "BF16", cases[i].sparseMode, cases[i].qT, cases[i].kvT,
                  cases[i].n1, cases[i].g, cases[i].d, cases[i].regionSize, cases[i].nLiveRegions,
                  cases[i].kvSeqlen, cases[i].fillMode, cases[i].reverseSlots, cases[i].inlineSentinels);
        int cRet = RunCase(stream, cases[i]);
        if (cRet != 0) {
            ++failed;
        }
    }

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    LOG_PRINT("summary: %d/%zu cases failed\n", failed, cases.size());
    return failed == 0 ? 0 : 1;
}
