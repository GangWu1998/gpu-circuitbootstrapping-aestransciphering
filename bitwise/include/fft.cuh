#pragma once

#include <cuda/std/complex>

#include <cufftdx.hpp>

#include "phantom.h"

namespace phantom::bitwise
{
    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
        requires std::is_same_v<RealType, __half> || std::is_same_v<RealType, float> || std::is_same_v<RealType, double>
    class cuFFTDxWrapper
    {
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

        cuFFTDxWrapper();

        void i2c_forward(complex_t* out, const BasicInteger* in, size_t batch_size,
                         const cudaStream_t& stream = cudaStreamPerThread);

        void c2i_inverse(BasicInteger* out, const complex_t* in, BasicInteger Q, size_t batch_size,
                               const cudaStream_t& stream);

        void multiply_and_accumulate(complex_t* acc,
                                     const complex_t* multiplicand,
                                     const complex_t* multiplier,
                                     size_t batch_size,
                                     const cudaStream_t& stream = cudaStreamPerThread);

    private:
        util::cuda_auto_ptr<complex_t> twist_table_;
    };
}
