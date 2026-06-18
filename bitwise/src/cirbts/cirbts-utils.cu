/*
 * =============================================================================
 * File: cirbts-utils.cu
 * Purpose: Utility kernels and helpers used by the GPU CirBTS pipeline,
 *          such as auto-map generation and special modulus switching.
 * Key parameters:
 *   - N/logn and modulus sizes that define map dimensions and scaling.
 *   - CUDA stream passed to helper kernels.
 * Key points:
 *   - Small, shared utilities used by init and bootstrapping stages.
 *   - Keeps helper logic isolated from the main pipeline code.
 * =============================================================================
 */
#include <cassert>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <utility>
#include <cuda_runtime_api.h>
#include "openfhe.h"
#include "cirbts/cirbts.cuh"
#include "cirbtscontext.h"
#include "cirbts/kernel.cuh"
#include "ntt.cuh"
#include "rlwe-homtrace.h"
#include "math/nbtheory.h"
#include "host/uintarithsmallmod.h"

using namespace lbcrypto;

void GPUCirBTSContext::GenerateAllAutoMaps(uint32_t* d_AllMaps, RLWECryptoParams* params, cudaStream_t s) const {
    const uint32_t N = params->GetN();
    const uint32_t m_mask = (N << 1) - 1;

    uint32_t logn = 0;
    for (uint32_t tmp = N; tmp > 1; tmp >>= 1) {
        ++logn;
    }
    const uint32_t numAuto = logn;

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x, numAuto);
    kernel_GenerateCoefficientAutoMaps<<<grid, block, 0, s>>>(d_AllMaps, N, logn, m_mask, numAuto);
}

NativeInteger GPUCirBTSContext::gpu_SpecilMS(const NativeInteger& v, const NativeInteger& q,
                                            const NativeInteger& Q, uint64_t bitwidth) const {
    NativeInteger v_ms =
        NativeInteger(static_cast<BasicInteger>(
                          std::floor(0.5 + v.ConvertToDouble() * q.ConvertToDouble() /
                                              (Q.ConvertToDouble() * static_cast<double>(1ULL << bitwidth))) *
                          static_cast<double>(1ULL << bitwidth)))
            .Mod(q);
    return v_ms;
}

bool GPUCirBTSContext::MaybeEnableL2Persist(const void* base_ptr, size_t bytes, cudaStream_t s) const {
    if (!l2_persist_enabled_ || l2_persist_failed_ || !base_ptr || bytes == 0 || l2_persist_bytes_ == 0) {
        return false;
    }

    const size_t window_bytes = std::min(bytes, l2_persist_bytes_);
    if (window_bytes == 0) {
        return false;
    }

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(base_ptr);
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio = std::max(0.0f, std::min(1.0f, l2_persist_hit_ratio_));
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    const cudaError_t err = cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    if (err != cudaSuccess) {
        l2_persist_failed_ = true;
        return false;
    }

    return true;
}

void GPUCirBTSContext::DisableL2Persist(cudaStream_t s) const {
    if (!l2_persist_enabled_ || l2_persist_failed_) {
        return;
    }
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio = 0.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
}
