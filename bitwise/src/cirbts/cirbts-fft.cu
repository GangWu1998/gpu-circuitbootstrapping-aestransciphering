/*
 * =============================================================================
 * File: cirbts-fft.cu
 * Purpose: cuFFTDx wrapper implementation and split-FFT helper kernels.
 * Key points:
 *   - Uses c2c block FFT with twist/untwist for negacyclic polynomials.
 *   - Converts integer polynomials to complex spectra and back modulo Q.
 * =============================================================================
 */
#include "cirbts/cirbts-fft.cuh"

#if defined(PHANTOM_ENABLE_CUFFTDX)

#include <cmath>

using namespace lbcrypto;

namespace {
    constexpr double kPi = 3.1415926535897932384626433832795028842;

    template <class Forward_FFT, typename RealType, typename ComplexType = cuda::std::complex<RealType>>
    __launch_bounds__(Forward_FFT::max_threads_per_block) __global__
    void forward_fft_kernel(ComplexType* output, const int64_t* input, const ComplexType* twist_table,
                            const size_t batch_size) {
        using real_t = RealType;
        using complex_t = ComplexType;

        complex_t thread_data[Forward_FFT::storage_size];

        const unsigned int local_fft_id = threadIdx.y;
        const unsigned int global_fft_id = blockIdx.x * Forward_FFT::ffts_per_block + local_fft_id;
        if (global_fft_id >= batch_size) {
            return;
        }
        const unsigned int offset = 2 * Forward_FFT::input_length * global_fft_id;
        const unsigned int stride = Forward_FFT::stride;
        unsigned int index = offset + threadIdx.x;

        for (unsigned int i = 0; i < Forward_FFT::input_ept; i++) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Forward_FFT::input_length) {
                const int64_t re = input[index];
                const int64_t im = input[index + Forward_FFT::input_length];
                thread_data[i] = complex_t(static_cast<real_t>(re), static_cast<real_t>(im));
                thread_data[i] *= twist_table[j];
                index += stride;
            }
        }

        extern __shared__ __align__(alignof(float4)) complex_t shared_mem[];
        Forward_FFT().execute(thread_data, shared_mem);

        const unsigned int out_offset = Forward_FFT::output_length * global_fft_id;
        index = out_offset + threadIdx.x;
        for (unsigned int i = 0; i < Forward_FFT::output_ept; ++i) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Forward_FFT::output_length) {
                output[index] = thread_data[i];
                index += stride;
            }
        }
    }

    template <class Forward_FFT, typename RealType, typename ComplexType = cuda::std::complex<RealType>>
    __launch_bounds__(Forward_FFT::max_threads_per_block) __global__
    void forward_fft_kernel_scaled(ComplexType* output, const int64_t* input, const ComplexType* twist_table,
                                   const RealType scale, const size_t batch_size) {
        using real_t = RealType;
        using complex_t = ComplexType;

        complex_t thread_data[Forward_FFT::storage_size];

        const unsigned int local_fft_id = threadIdx.y;
        const unsigned int global_fft_id = blockIdx.x * Forward_FFT::ffts_per_block + local_fft_id;
        if (global_fft_id >= batch_size) {
            return;
        }
        const unsigned int offset = 2 * Forward_FFT::input_length * global_fft_id;
        const unsigned int stride = Forward_FFT::stride;
        unsigned int index = offset + threadIdx.x;

        for (unsigned int i = 0; i < Forward_FFT::input_ept; i++) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Forward_FFT::input_length) {
                const int64_t re = input[index];
                const int64_t im = input[index + Forward_FFT::input_length];
                thread_data[i] = complex_t(static_cast<real_t>(re) * scale, static_cast<real_t>(im) * scale);
                thread_data[i] *= twist_table[j];
                index += stride;
            }
        }

        extern __shared__ __align__(alignof(float4)) complex_t shared_mem[];
        Forward_FFT().execute(thread_data, shared_mem);

        const unsigned int out_offset = Forward_FFT::output_length * global_fft_id;
        index = out_offset + threadIdx.x;
        for (unsigned int i = 0; i < Forward_FFT::output_ept; ++i) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Forward_FFT::output_length) {
                output[index] = thread_data[i];
                index += stride;
            }
        }
    }

    template <class Inverse_FFT, typename RealType, typename ComplexType = cuda::std::complex<RealType>>
    __launch_bounds__(Inverse_FFT::max_threads_per_block) __global__
    void inverse_fft_kernel(BasicInteger* output, const ComplexType* input,
                            const ComplexType* twist_table, const int64_t Q, const size_t batch_size) {
        using real_t = RealType;
        using complex_t = ComplexType;

        complex_t thread_data[Inverse_FFT::storage_size];

        const unsigned int local_fft_id = threadIdx.y;
        const unsigned int global_fft_id = blockIdx.x * Inverse_FFT::ffts_per_block + local_fft_id;
        if (global_fft_id >= batch_size) {
            return;
        }
        const unsigned int offset = Inverse_FFT::input_length * global_fft_id;
        const unsigned int stride = Inverse_FFT::stride;
        unsigned int index = offset + threadIdx.x;

        for (unsigned int i = 0; i < Inverse_FFT::input_ept; i++) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Inverse_FFT::input_length) {
                thread_data[i] = input[index];
                index += stride;
            }
        }

        extern __shared__ __align__(alignof(float4)) complex_t shared_mem[];
        Inverse_FFT().execute(thread_data, shared_mem);

        const unsigned int out_offset = 2 * Inverse_FFT::output_length * global_fft_id;
        index = out_offset + threadIdx.x;
        for (unsigned int i = 0; i < Inverse_FFT::output_ept; ++i) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Inverse_FFT::output_length) {
                constexpr real_t inv_n = 1.0 / cufftdx::size_of<Inverse_FFT>::value;
                thread_data[i] *= inv_n;
                thread_data[i] *= complex_t(twist_table[j].real(), -twist_table[j].imag());

                const int64_t a_round = static_cast<int64_t>(llround(thread_data[i].real()));
                const int64_t b_round = static_cast<int64_t>(llround(thread_data[i].imag()));
                int64_t a = a_round % Q;
                int64_t b = b_round % Q;
                if (a < 0) a += Q;
                if (b < 0) b += Q;

                output[index] = static_cast<BasicInteger>(a);
                output[index + Inverse_FFT::output_length] = static_cast<BasicInteger>(b);
                index += stride;
            }
        }
    }

    template <class Inverse_FFT, typename RealType, typename ComplexType = cuda::std::complex<RealType>>
    __launch_bounds__(Inverse_FFT::max_threads_per_block) __global__
    void inverse_fft_kernel_scale_add(BasicInteger* acc, const ComplexType* input, const ComplexType* twist_table,
                                      const int64_t Q, const uint64_t scale, const uint64_t scale_shoup,
                                      const size_t batch_size) {
        using real_t = RealType;
        using complex_t = ComplexType;

        complex_t thread_data[Inverse_FFT::storage_size];

        const unsigned int local_fft_id = threadIdx.y;
        const unsigned int global_fft_id = blockIdx.x * Inverse_FFT::ffts_per_block + local_fft_id;
        if (global_fft_id >= batch_size) {
            return;
        }
        const unsigned int offset = Inverse_FFT::input_length * global_fft_id;
        const unsigned int stride = Inverse_FFT::stride;
        unsigned int index = offset + threadIdx.x;

        for (unsigned int i = 0; i < Inverse_FFT::input_ept; i++) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Inverse_FFT::input_length) {
                thread_data[i] = input[index];
                index += stride;
            }
        }

        extern __shared__ __align__(alignof(float4)) complex_t shared_mem[];
        Inverse_FFT().execute(thread_data, shared_mem);

        const unsigned int out_offset = 2 * Inverse_FFT::output_length * global_fft_id;
        index = out_offset + threadIdx.x;
        for (unsigned int i = 0; i < Inverse_FFT::output_ept; ++i) {
            const size_t j = i * stride + threadIdx.x;
            if (j < Inverse_FFT::output_length) {
                constexpr real_t inv_n = 1.0 / cufftdx::size_of<Inverse_FFT>::value;
                thread_data[i] *= inv_n;
                thread_data[i] *= complex_t(twist_table[j].real(), -twist_table[j].imag());

                const int64_t a_round = static_cast<int64_t>(llround(thread_data[i].real()));
                const int64_t b_round = static_cast<int64_t>(llround(thread_data[i].imag()));
                int64_t a = a_round % Q;
                int64_t b = b_round % Q;
                if (a < 0) a += Q;
                if (b < 0) b += Q;

                uint64_t a_u = static_cast<uint64_t>(a);
                uint64_t b_u = static_cast<uint64_t>(b);
                if (scale != 1) {
                    a_u = phantom::arith::multiply_and_reduce_shoup(a_u, scale, scale_shoup, static_cast<uint64_t>(Q));
                    b_u = phantom::arith::multiply_and_reduce_shoup(b_u, scale, scale_shoup, static_cast<uint64_t>(Q));
                }

                uint64_t out0 = acc[index] + a_u;
                if (out0 >= static_cast<uint64_t>(Q)) {
                    out0 -= static_cast<uint64_t>(Q);
                }
                acc[index] = static_cast<BasicInteger>(out0);

                uint64_t out1 = acc[index + Inverse_FFT::output_length] + b_u;
                if (out1 >= static_cast<uint64_t>(Q)) {
                    out1 -= static_cast<uint64_t>(Q);
                }
                acc[index + Inverse_FFT::output_length] = static_cast<BasicInteger>(out1);
                index += stride;
            }
        }
    }

} // namespace

namespace phantom::bitwise {
    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
    cuFFTDxWrapperCirBTS<RealType, PolyDim, FFTsPerBlock, ElementsPerThread, Arch>::cuFFTDxWrapperCirBTS() {
        const size_t twist_size = PolyDim / 2;
        twist_table_ = phantom::util::make_cuda_auto_ptr<complex_t>(twist_size, cudaStreamPerThread);

        std::vector<complex_t> host_twist(twist_size);
        const double angle = kPi / static_cast<double>(PolyDim);
        for (size_t i = 0; i < twist_size; ++i) {
            const double theta = angle * static_cast<double>(i);
            host_twist[i] = complex_t(static_cast<real_t>(std::cos(theta)), static_cast<real_t>(std::sin(theta)));
        }
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(twist_table_.get(), host_twist.data(),
                                           twist_size * sizeof(complex_t),
                                           cudaMemcpyHostToDevice, cudaStreamPerThread));
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));
    }

    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
    void cuFFTDxWrapperCirBTS<RealType, PolyDim, FFTsPerBlock, ElementsPerThread, Arch>::i2c_forward(
        complex_t* out, const int64_t* in, const size_t batch_size, const cudaStream_t& stream) {
        if (batch_size == 0) {
            return;
        }
        constexpr unsigned int ffts_per_block = Forward_FFT::ffts_per_block;
        const unsigned int grid_x = static_cast<unsigned int>((batch_size + ffts_per_block - 1) / ffts_per_block);
        const dim3 grid_dim(grid_x);
        constexpr dim3 block_dim = Forward_FFT::block_dim;
        constexpr size_t shared_bytes = Forward_FFT::shared_memory_size;
        forward_fft_kernel<Forward_FFT, RealType>
            <<<grid_dim, block_dim, shared_bytes, stream>>>(out, in, twist_table_.get(), batch_size);
        PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
    }

    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
    void cuFFTDxWrapperCirBTS<RealType, PolyDim, FFTsPerBlock, ElementsPerThread, Arch>::i2c_forward_scaled(
        complex_t* out, const int64_t* in, const double scale, const size_t batch_size, const cudaStream_t& stream) {
        if (batch_size == 0) {
            return;
        }
        constexpr unsigned int ffts_per_block = Forward_FFT::ffts_per_block;
        const unsigned int grid_x = static_cast<unsigned int>((batch_size + ffts_per_block - 1) / ffts_per_block);
        const dim3 grid_dim(grid_x);
        constexpr dim3 block_dim = Forward_FFT::block_dim;
        constexpr size_t shared_bytes = Forward_FFT::shared_memory_size;
        forward_fft_kernel_scaled<Forward_FFT, RealType>
            <<<grid_dim, block_dim, shared_bytes, stream>>>(out, in, twist_table_.get(), static_cast<RealType>(scale),
                                                            batch_size);
        PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
    }

    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
    void cuFFTDxWrapperCirBTS<RealType, PolyDim, FFTsPerBlock, ElementsPerThread, Arch>::c2i_inverse(
        BasicInteger* out, const complex_t* in, const BasicInteger Q, const size_t batch_size,
        const cudaStream_t& stream) {
        if (batch_size == 0) {
            return;
        }
        constexpr unsigned int ffts_per_block = Inverse_FFT::ffts_per_block;
        const unsigned int grid_x = static_cast<unsigned int>((batch_size + ffts_per_block - 1) / ffts_per_block);
        const dim3 grid_dim(grid_x);
        constexpr dim3 block_dim = Inverse_FFT::block_dim;
        constexpr size_t shared_bytes = Inverse_FFT::shared_memory_size;
        inverse_fft_kernel<Inverse_FFT, RealType>
            <<<grid_dim, block_dim, shared_bytes, stream>>>(out, in, twist_table_.get(), static_cast<int64_t>(Q),
                                                            batch_size);
        PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
    }

    template <typename RealType,
              unsigned int PolyDim,
              unsigned int FFTsPerBlock,
              unsigned int ElementsPerThread,
              unsigned int Arch>
    void cuFFTDxWrapperCirBTS<RealType, PolyDim, FFTsPerBlock, ElementsPerThread, Arch>::c2i_inverse_add(
        BasicInteger* acc, const complex_t* in, const BasicInteger Q, const BasicInteger scale,
        const BasicInteger scale_shoup, const size_t batch_size, const cudaStream_t& stream) {
        if (batch_size == 0) {
            return;
        }
        constexpr unsigned int ffts_per_block = Inverse_FFT::ffts_per_block;
        const unsigned int grid_x = static_cast<unsigned int>((batch_size + ffts_per_block - 1) / ffts_per_block);
        const dim3 grid_dim(grid_x);
        constexpr dim3 block_dim = Inverse_FFT::block_dim;
        constexpr size_t shared_bytes = Inverse_FFT::shared_memory_size;
        inverse_fft_kernel_scale_add<Inverse_FFT, RealType>
            <<<grid_dim, block_dim, shared_bytes, stream>>>(
                acc, in, twist_table_.get(), static_cast<int64_t>(Q), static_cast<uint64_t>(scale),
                static_cast<uint64_t>(scale_shoup), batch_size);
        PHANTOM_CHECK_CUDA(cudaPeekAtLastError());
    }

    template class cuFFTDxWrapperCirBTS<double, 1024, 1, 16, PHANTOM_CUFFTDX_SM>;
    template class cuFFTDxWrapperCirBTS<double, 2048, 1, 16, PHANTOM_CUFFTDX_SM>;
    template class cuFFTDxWrapperCirBTS<double, 2048, 2, 16, PHANTOM_CUFFTDX_SM>;
    template class cuFFTDxWrapperCirBTS<double, 2048, 4, 16, PHANTOM_CUFFTDX_SM>;
    template class cuFFTDxWrapperCirBTS<double, 4096, 1, 16, PHANTOM_CUFFTDX_SM>;
} // namespace phantom::bitwise

__global__ void kernel_ExtractKeyLimb(int64_t* out, const uint64_t* in, const size_t count, const uint32_t limb_bits,
                                      const uint32_t limb_idx) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const uint32_t shift = limb_bits * limb_idx;
    const uint64_t mask = (limb_bits >= 64u) ? ~0ull : ((1ull << limb_bits) - 1ull);
    const uint64_t limb = (in[idx] >> shift) & mask;
    out[idx] = static_cast<int64_t>(limb);
}

__global__ void kernel_EvalAccCoreCGGI_fft_monic(SplitFFTComplex* acc, const SplitFFTComplex* dct,
                                                 const SplitFFTComplex* key, const SplitFFTComplex* monic_fft,
                                                 const size_t fftN, const uint32_t digitsG2,
                                                 const uint32_t blocks_per_poly, const uint32_t* indexPos,
                                                 const uint32_t lweIndex) {
    const uint32_t poly_idx = blockIdx.x / blocks_per_poly;
    if (poly_idx >= 2) {
        return;
    }
    const uint32_t block_in_poly = blockIdx.x - poly_idx * blocks_per_poly;
    __shared__ uint32_t shared_index;
    if (threadIdx.x == 0) {
        shared_index = indexPos[lweIndex];
    }
    __syncthreads();

    const SplitFFTComplex* monic = monic_fft + static_cast<size_t>(shared_index) * fftN;
    const size_t N_idx = static_cast<size_t>(block_in_poly) * blockDim.x + threadIdx.x;
    if (N_idx >= fftN) {
        return;
    }
    SplitFFTComplex acc_tmp{0.0, 0.0};
    for (uint32_t d = 0; d < digitsG2; ++d) {
        const SplitFFTComplex dct_coeff = dct[static_cast<size_t>(d) * fftN + N_idx];
        const SplitFFTComplex key_coeff = key[static_cast<size_t>(d) * 2 * fftN + poly_idx * fftN + N_idx];
        acc_tmp += dct_coeff * key_coeff;
    }
    acc[poly_idx * fftN + N_idx] = acc_tmp * monic[N_idx];
}

#endif  // PHANTOM_ENABLE_CUFFTDX
