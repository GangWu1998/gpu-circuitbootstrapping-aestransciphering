/*
 * =============================================================================
 * File: cirbts-fft.cuh
 * Purpose: cuFFTDx wrapper and split-FFT helper kernels for CGGI EvalAcc.
 * Key parameters:
 *   - PHANTOM_CUFFTDX_SM: target SM for cuFFTDx (default 900 for SM90).
 *   - PolyDim: polynomial degree (N), FFT length is N/2.
 * Key points:
 *   - Uses twist table exp(i*pi*j/N) for negacyclic convolution.
 *   - Provides kernels to center digits, extract key limbs, and apply monic update.
 * =============================================================================
 */
#pragma once

#include <cuda/std/complex>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "phantom.h"

#ifndef PHANTOM_CUFFTDX_SM
#define PHANTOM_CUFFTDX_SM 900
#endif

#if defined(PHANTOM_ENABLE_CUFFTDX)
#include <cufftdx.hpp>

namespace phantom::bitwise {
    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
    class cuFFTDxWrapperCirBTS {
    public:
        using Forward_FFT = decltype(cufftdx::Block() +
            cufftdx::Size<PolyDim / 2>() +
            cufftdx::Precision<RealType>() +
            cufftdx::ElementsPerThread<ElementsPerThread>() +
            cufftdx::FFTsPerBlock<FFTsPerBlock>() +
            cufftdx::SM<Arch>() +
            cufftdx::Type<cufftdx::fft_type::c2c>() +
            cufftdx::Direction<cufftdx::fft_direction::forward>());

        using Inverse_FFT = decltype(cufftdx::Block() +
            cufftdx::Size<PolyDim / 2>() +
            cufftdx::Precision<RealType>() +
            cufftdx::ElementsPerThread<ElementsPerThread>() +
            cufftdx::FFTsPerBlock<FFTsPerBlock>() +
            cufftdx::SM<Arch>() +
            cufftdx::Type<cufftdx::fft_type::c2c>() +
            cufftdx::Direction<cufftdx::fft_direction::inverse>());

        using real_t = RealType;
        using complex_t = cuda::std::complex<RealType>;

        cuFFTDxWrapperCirBTS();

        void i2c_forward(complex_t* out, const int64_t* in, size_t batch_size,
                         const cudaStream_t& stream = cudaStreamPerThread);
        void i2c_forward_scaled(complex_t* out, const int64_t* in, double scale, size_t batch_size,
                                const cudaStream_t& stream = cudaStreamPerThread);

        void c2i_inverse(BasicInteger* out, const complex_t* in, BasicInteger Q, size_t batch_size,
                         const cudaStream_t& stream = cudaStreamPerThread);
        void c2i_inverse_add(BasicInteger* acc, const complex_t* in, BasicInteger Q, BasicInteger scale,
                             BasicInteger scale_shoup, size_t batch_size,
                             const cudaStream_t& stream = cudaStreamPerThread);

    private:
        phantom::util::cuda_auto_ptr<complex_t> twist_table_;
    };

    using SplitFFTComplex = cuda::std::complex<double>;
}
#else
namespace phantom::bitwise {
    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
    class cuFFTDxWrapperCirBTS {
    public:
        using real_t = RealType;
        using complex_t = cuda::std::complex<RealType>;

        cuFFTDxWrapperCirBTS() = default;

        void i2c_forward(complex_t*, const int64_t*, size_t,
                         const cudaStream_t& = cudaStreamPerThread) {}
        void i2c_forward_scaled(complex_t*, const int64_t*, double, size_t,
                                const cudaStream_t& = cudaStreamPerThread) {}

        void c2i_inverse(BasicInteger*, const complex_t*, BasicInteger, size_t,
                         const cudaStream_t& = cudaStreamPerThread) {}
        void c2i_inverse_add(BasicInteger*, const complex_t*, BasicInteger, BasicInteger, BasicInteger, size_t,
                             const cudaStream_t& = cudaStreamPerThread) {}
    };

    using SplitFFTComplex = cuda::std::complex<double>;
}
#endif

using phantom::bitwise::SplitFFTComplex;

__global__ void kernel_ExtractKeyLimb(int64_t* out, const uint64_t* in, size_t count, uint32_t limb_bits,
                                      uint32_t limb_idx);
__global__ void kernel_EvalAccCoreCGGI_fft_monic(SplitFFTComplex* acc, const SplitFFTComplex* dct,
                                                 const SplitFFTComplex* key, const SplitFFTComplex* monic_fft,
                                                 size_t fftN, uint32_t digitsG2, uint32_t blocks_per_poly,
                                                 const uint32_t* indexPos, uint32_t lweIndex);
