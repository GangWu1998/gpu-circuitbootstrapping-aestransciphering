#pragma once

#include "openfhe.h"
#include "phantom.h"

#include "compact_ntt.cuh"
#include "fft.cuh"

#ifndef PHANTOM_CUFFTDX_SM
#define PHANTOM_CUFFTDX_SM 900
#endif

namespace phantom::bitwise
{
    using real_t = double;
    using complex_t = cuda::std::complex<real_t>;

    typedef std::vector<std::vector<std::vector<phantom::util::cuda_auto_ptr<BasicInteger>>>> GPURingGSWACCKey;
    typedef std::vector<std::vector<std::vector<phantom::util::cuda_auto_ptr<complex_t>>>> GPURingGSWACCKey_fft;

    class GPURingGSWBTKey
    {
    public:
        // refreshing key
        GPURingGSWACCKey RGSWACCKey;
        GPURingGSWACCKey_fft RGSWACCKey_fft;

        // switching key
        phantom::util::cuda_auto_ptr<BasicInteger> LWESwitchKey_A;
        std::vector<std::vector<std::vector<NativeInteger>>> cpu_keyB;
    };

    class GPURingGSWAccumulator : public lbcrypto::RingGSWAccumulator
    {
    private:
        void GPUKeyGenAccDM(GPURingGSWBTKey& gpu_ek,
                            const std::shared_ptr<lbcrypto::RingGSWCryptoParams>& params,
                            const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                            lbcrypto::ConstLWEPrivateKey& LWEsk,
                            const cudaStream_t& stream) const;

        void GPUKeyGenAccCGGI(GPURingGSWBTKey& gpu_ek,
                              const std::shared_ptr<lbcrypto::RingGSWCryptoParams>& params,
                              const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                              lbcrypto::ConstLWEPrivateKey& LWEsk,
                              const cudaStream_t& stream) const;

        void GPUKeyGenDM(phantom::util::cuda_auto_ptr<BasicInteger>& d_keyNTT,
                         phantom::util::cuda_auto_ptr<complex_t>& d_keyFFT,
                         const std::shared_ptr<lbcrypto::RingGSWCryptoParams>& params,
                         const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                         lbcrypto::LWEPlaintext m,
                         const cudaStream_t& stream) const;

        void GPUKeyGenCGGI(phantom::util::cuda_auto_ptr<BasicInteger>& d_keyNTT,
                           phantom::util::cuda_auto_ptr<complex_t>& d_keyFFT,
                           const std::shared_ptr<lbcrypto::RingGSWCryptoParams>& params,
                           const phantom::util::cuda_auto_ptr<BasicInteger>& d_skNTT,
                           lbcrypto::LWEPlaintext m,
                           const cudaStream_t& stream) const;

        void GPUEvalAccDM(const std::shared_ptr<lbcrypto::BinFHECryptoParams>& params, const GPURingGSWBTKey& EK,
                          const NativeVector& a, const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                          const cudaStream_t& s, bool use_fft = false) const;

        void BatchGPUEvalAccDM(const std::shared_ptr<lbcrypto::BinFHECryptoParams>& params, const GPURingGSWBTKey& EK,
                               const std::vector<NativeVector>& v_a,
                               const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                               const cudaStream_t& s, bool use_fft = false) const;

        void GPUEvalAccCGGI(const std::shared_ptr<lbcrypto::BinFHECryptoParams>& params, const GPURingGSWBTKey& EK,
                            const NativeVector& a, const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                            const cudaStream_t& s, bool use_fft = false) const;

        void BatchGPUCGGI(const std::shared_ptr<lbcrypto::BinFHECryptoParams>& params, const GPURingGSWBTKey& EK,
                                 const std::vector<NativeVector>& v_a,
                                 const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                                 const cudaStream_t& s, bool use_fft = false) const;

    protected:
        bool use_fft_ = false; // use NTT by default

        // for NTT
        phantom::util::cuda_auto_ptr<BasicInteger> d_monic_polys_; // for GINX
        std::shared_ptr<FourStepNTT> ntt_;

        // for FFT
        phantom::util::cuda_auto_ptr<complex_t> d_monic_polys_fft_; // for GINX
        std::shared_ptr<cuFFTDxWrapper<double, 1024, 1, 16, PHANTOM_CUFFTDX_SM>> fft_1024_;
        std::shared_ptr<cuFFTDxWrapper<double, 2048, 1, 16, PHANTOM_CUFFTDX_SM>> fft_2048_;
        std::shared_ptr<cuFFTDxWrapper<double, 4096, 1, 16, PHANTOM_CUFFTDX_SM>> fft_4096_;

    public:
        GPURingGSWAccumulator() = default;

        explicit GPURingGSWAccumulator(const bool use_fft)
        {
            if (use_fft)
            {
                use_fft_ = true;
                fft_1024_ = std::make_shared<cuFFTDxWrapper<double, 1024, 1, 16, PHANTOM_CUFFTDX_SM>>();
                fft_2048_ = std::make_shared<cuFFTDxWrapper<double, 2048, 1, 16, PHANTOM_CUFFTDX_SM>>();
                fft_4096_ = std::make_shared<cuFFTDxWrapper<double, 4096, 1, 16, PHANTOM_CUFFTDX_SM>>();
            }
        }

        /**
         * Key generation for internal Ring GSW
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param sk secret key polynomial in the COEFFICIENT representation
         * @param LWEsk the secret key
         * @param stream CUDA stream
         * @return a shared pointer to the resulting keys
         */
        void GPUKeyGenAcc(GPURingGSWBTKey& gpu_ek,
                          const std::shared_ptr<lbcrypto::RingGSWCryptoParams>& params,
                          const lbcrypto::NativeVector& sk, lbcrypto::ConstLWEPrivateKey& LWEsk,
                          const cudaStream_t& stream);

        /**
       * Main accumulator function used in bootstrapping
       *
       * @param params a shared pointer to RingGSW scheme parameters
       * @param ek the RGSW bootstrapping key
       * @param a ciphertext
       * @param acc previous value of the accumulator
       * @param s CUDA stream
       * @param use_fft
       */
        void GPUEvalAcc(const std::shared_ptr<lbcrypto::BinFHECryptoParams>& params, const GPURingGSWBTKey& EK,
                        const NativeVector& a, const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                        bool use_fft,
                        const cudaStream_t& s) const;

        void BatchGPUEvalAcc(const std::shared_ptr<lbcrypto::BinFHECryptoParams>& params, const GPURingGSWBTKey& EK,
                             const std::vector<NativeVector>& v_a,
                             const phantom::util::cuda_auto_ptr<BasicInteger>& d_acc,
                             bool use_fft,
                             const cudaStream_t& s) const;
    };
} // namespace lbcrypto
