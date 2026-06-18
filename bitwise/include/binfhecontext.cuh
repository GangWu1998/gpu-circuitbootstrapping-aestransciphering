#pragma once

#include "openfhe.h"
#include "phantom.h"
#include "binfhe-base-scheme.cuh"

namespace phantom::bitwise {
    class GPUBinFHEContext : public lbcrypto::BinFHEContext {
        std::shared_ptr<lbcrypto::BinFHECryptoParams> params_{nullptr};

        // Shared pointer to the underlying RingGSW/RLWE scheme
        std::shared_ptr<GPUBinFHEScheme> gpu_binfhescheme_{nullptr};

        // Struct containing the bootstrapping keys
        std::map<uint32_t, GPURingGSWBTKey> d_BTKey_map_;

    public:
        explicit GPUBinFHEContext(BinFHEContext& cc) {
            // GPU setup
            int device;
            CUDA_CHECK_AND_EXIT(cudaGetDevice(&device))
            cudaMemPool_t mempool;
            CUDA_CHECK_AND_EXIT(cudaDeviceGetDefaultMemPool(&mempool, device))
            uint64_t threshold = UINT64_MAX;
            CUDA_CHECK_AND_EXIT(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold))

            params_ = cc.GetParams();
            gpu_binfhescheme_ = std::make_shared<GPUBinFHEScheme>(params_, cudaStreamPerThread);
        }

        /**
         * Generates bootstrapping keys
         *
         * @param sk secret key
         * @param keygenMode key generation mode for symmetric or public encryption
         */
        void GPUBTKeyGen(lbcrypto::ConstLWEPrivateKey& sk, lbcrypto::KEYGEN_MODE keygenMode = lbcrypto::SYM_ENCRYPT);

        /**
         * Evaluates a binary gate (calls bootstrapping as a subroutine)
         *
         * @param gate the gate; can be AND, OR, NAND, NOR, XOR, or XNOR
         * @param ct1 first ciphertext
         * @param ct2 second ciphertext
         * @param stream CUDA stream
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalBinGate(lbcrypto::BINGATE gate,
                       lbcrypto::ConstLWECiphertext& ct1, lbcrypto::ConstLWECiphertext& ct2,
                       const cudaStream_t& stream = cudaStreamPerThread) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalBinGate(lbcrypto::BINGATE gate,
                            const std::vector<lbcrypto::LWECiphertext>& v_ct1,
                            const std::vector<lbcrypto::LWECiphertext>& v_ct2,
                            const cudaStream_t& stream = cudaStreamPerThread) const;

        /**
         * Evaluate an arbitrary function
         *
         * @param ct ciphertext to be bootstrapped
         * @param LUT the look-up table of the to-be-evaluated function
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalFunc(lbcrypto::ConstLWECiphertext& ct, const std::vector<NativeInteger>& LUT,
                    const cudaStream_t& stream = cudaStreamPerThread) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalFunc(const std::vector<lbcrypto::LWECiphertext>& v_ct, const std::vector<NativeInteger>& LUT,
                         const cudaStream_t& stream = cudaStreamPerThread) const;

        /**
         * Evaluate a round down function
         *
         * @param ct ciphertext to be bootstrapped
         * @param roundbits number of bits to be rounded
         * @param stream CUDA stream
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalFloor(lbcrypto::ConstLWECiphertext& ct, uint32_t roundbits = 0,
                     const cudaStream_t& stream = cudaStreamPerThread) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalFloor(const std::vector<lbcrypto::LWECiphertext>& v_ct, uint32_t roundbits = 0,
                          const cudaStream_t& stream = cudaStreamPerThread) const;

        /**
         * Evaluate a sign function over large precisions
         *
         * @param ct ciphertext to be bootstrapped
         * @param schemeSwitch flag that indicates if it should be compatible to scheme switching
         * @param stream CUDA stream
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalSign(lbcrypto::ConstLWECiphertext& ct, bool schemeSwitch = false,
                    const cudaStream_t& stream = cudaStreamPerThread) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalSign(const std::vector<lbcrypto::LWECiphertext>& v_ct, bool schemeSwitch = false,
                         const cudaStream_t& stream = cudaStreamPerThread) const;

        /**
         * Evaluate ciphertext decomposition
         *
         * @param ct ciphertext to be bootstrapped
         * @param stream CUDA stream
         * @return a vector of shared pointers to the resulting ciphertexts
         */
        std::vector<lbcrypto::LWECiphertext>
        GPUEvalDecomp(lbcrypto::ConstLWECiphertext& ct, const cudaStream_t& stream = cudaStreamPerThread) const;

        std::vector<std::vector<lbcrypto::LWECiphertext>>
        BatchGPUEvalDecomp(const std::vector<lbcrypto::LWECiphertext>& v_ct,
                           const cudaStream_t& stream = cudaStreamPerThread) const;
    };
}
