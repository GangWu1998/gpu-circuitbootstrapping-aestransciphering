#include <cassert>
#include "openfhe.h"
#include "binfhecontext.cuh"
#include "kernel.cuh"
#include "ntt.cuh"

using namespace lbcrypto;

namespace phantom::bitwise {
    void GPUBinFHEContext::GPUBTKeyGen(ConstLWEPrivateKey& sk, const KEYGEN_MODE keygenMode) {
        if (d_BTKey_map_.empty()) {
            auto temp = params_->GetRingGSWParams()->GetBaseG();
            d_BTKey_map_[temp] = gpu_binfhescheme_->GPUKeyGen(params_, sk, cudaStreamPerThread, keygenMode);
        }
    }

    LWECiphertext GPUBinFHEContext::GPUEvalBinGate(const BINGATE gate, ConstLWECiphertext& ct1, ConstLWECiphertext& ct2,
                                                   const cudaStream_t& stream) const {
        const auto& EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->GPUEvalBinGate(params_, gate, EK, ct1, ct2, stream);
    }

    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalBinGate(const BINGATE gate,
                                                                     const std::vector<LWECiphertext>& v_ct1,
                                                                     const std::vector<LWECiphertext>& v_ct2,
                                                                     const cudaStream_t& stream) const {
        const auto& EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->BatchGPUEvalBinGate(params_, gate, EK, v_ct1, v_ct2, stream);
    }

    LWECiphertext GPUBinFHEContext::GPUEvalFunc(ConstLWECiphertext& ct, const std::vector<NativeInteger>& LUT,
                                                const cudaStream_t& stream) const {
        const auto& EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->GPUEvalFunc(params_, EK, ct, LUT, GetBeta(), stream);
    }

    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalFunc(const std::vector<LWECiphertext>& v_ct,
                                                                  const std::vector<NativeInteger>& LUT,
                                                                  const cudaStream_t& stream) const {
        const auto& EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->BatchGPUEvalFunc(params_, EK, v_ct, LUT, GetBeta(), stream);
    }

    LWECiphertext GPUBinFHEContext::GPUEvalFloor(ConstLWECiphertext& ct, const uint32_t roundbits,
                                                 const cudaStream_t& stream) const {
        const auto& EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->GPUEvalFloor(params_, EK, ct, GetBeta(), stream, roundbits);
    }

    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalFloor(const std::vector<LWECiphertext>& v_ct,
                                                                   const uint32_t roundbits,
                                                                   const cudaStream_t& stream) const {
        const auto& EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->BatchGPUEvalFloor(params_, EK, v_ct, GetBeta(), stream, roundbits);
    }

    LWECiphertext GPUBinFHEContext::GPUEvalSign(ConstLWECiphertext& ct, bool schemeSwitch,
                                                const cudaStream_t& stream) const {
        return gpu_binfhescheme_->GPUEvalSign(params_, d_BTKey_map_, ct, GetBeta(), stream, schemeSwitch);
    }

    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalSign(const std::vector<LWECiphertext>& v_ct,
                                                                  bool schemeSwitch,
                                                                  const cudaStream_t& stream) const {
        return gpu_binfhescheme_->BatchGPUEvalSign(params_, d_BTKey_map_, v_ct, GetBeta(), stream, schemeSwitch);
    }

    std::vector<LWECiphertext> GPUBinFHEContext::GPUEvalDecomp(ConstLWECiphertext& ct,
                                                               const cudaStream_t& stream) const {
        return gpu_binfhescheme_->GPUEvalDecomp(params_, d_BTKey_map_, ct, GetBeta(), stream);
    }

    std::vector<std::vector<LWECiphertext>> GPUBinFHEContext::BatchGPUEvalDecomp(const std::vector<LWECiphertext>& v_ct,
        const cudaStream_t& stream) const {
        return gpu_binfhescheme_->BatchGPUEvalDecomp(params_, d_BTKey_map_, v_ct, GetBeta(), stream);
    }
}
