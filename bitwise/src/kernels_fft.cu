#include "kernels_fft.cuh"
#include "modarith.cuh"

using namespace lbcrypto;

__global__ void kernel_EvalAccCoreDM_fft(complex_t* acc, const complex_t* dct, const complex_t* RingGSWACCKey,
                                         const size_t N, const size_t digitsG2)
{
    const size_t poly_offset = blockIdx.x * N;
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x)
    {
        complex_t acc_tmp{0.0, 0.0};

        for (size_t digitsG2_index = 0; digitsG2_index < digitsG2; ++digitsG2_index)
        {
            const complex_t dct_coeff = dct[digitsG2_index * N + N_idx];
            const complex_t key_coeff = RingGSWACCKey[digitsG2_index * 2 * N + poly_offset + N_idx];
            acc_tmp += dct_coeff * key_coeff;
        }

        acc[poly_offset + N_idx] = acc_tmp;
    }
}

__global__ void kernel_EvalAccCoreDM_fft_batch(complex_t* acc, const complex_t* dct, complex_t** acc_keys,
                                               const size_t N, const size_t digitsG2)
{
    const size_t poly_offset = blockIdx.x * N;
    const size_t batch_idx = blockIdx.y;
    const complex_t* acc_key = acc_keys[batch_idx];

    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x)
    {
        complex_t acc_tmp{0.0, 0.0};

        for (size_t digitsG2_idx = 0; digitsG2_idx < digitsG2; ++digitsG2_idx)
        {
            const complex_t dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_idx * N + N_idx];
            const complex_t key_coeff = acc_key[digitsG2_idx * 2 * N + poly_offset + N_idx];
            acc_tmp += dct_coeff * key_coeff;
        }

        acc[batch_idx * 2 * N + poly_offset + N_idx] = acc_tmp;
    }
}

__global__ void kernel_EvalAccCoreCGGI_fft(
    complex_t* acc, const complex_t* dct, const complex_t* d_ACCKey,
    const size_t N, const size_t digitsG2)
{
    const size_t poly_idx = blockIdx.x;
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x)
    {
        complex_t acc_tmp{0.0, 0.0};

        for (uint32_t digitsG2_idx = 0; digitsG2_idx < digitsG2; ++digitsG2_idx)
        {
            const complex_t dct_coeff = dct[digitsG2_idx * N + N_idx];
            const complex_t key_coeff = d_ACCKey[digitsG2_idx * 2 * N + poly_idx * N + N_idx];

            acc_tmp += dct_coeff * key_coeff;
        }

        acc[poly_idx * N + N_idx] = acc_tmp;
    }
}

__global__ void kernel_EvalAccCoreCGGI_fft_batch(
    complex_t* acc, const complex_t* dct, const complex_t* d_ACCKey,
    const size_t N, const size_t digitsG2)
{
    const size_t poly_offset = blockIdx.x * N;
    const size_t batch_idx = blockIdx.y;
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x)
    {
        complex_t acc_tmp{0.0, 0.0};

        for (uint32_t digitsG2_idx = 0; digitsG2_idx < digitsG2; ++digitsG2_idx)
        {
            const complex_t dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_idx * N + N_idx];
            const complex_t key_coeff = d_ACCKey[digitsG2_idx * 2 * N + poly_offset + N_idx];

            acc_tmp += dct_coeff * key_coeff;
        }

        acc[batch_idx * 2 * N + poly_offset + N_idx] = acc_tmp;
    }
}

__global__ void kernel_EvalAccCoreCGGI_mac_monic(BasicInteger* output, const BasicInteger* input,
                                                 const size_t N, const BasicInteger Q, const size_t m)
{
    const size_t poly_idx = blockIdx.x;
    const int k = m & (N - 1);
    const bool hi = (m >= N);
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x)
    {
        int idx = N_idx - k;
        const bool wrapped = (idx < 0);
        if (wrapped) idx += N;

        BasicInteger coeff = input[poly_idx * N + idx];
        const bool neg = hi ^ wrapped;

        if (neg && coeff)
            coeff = Q - coeff;

        BasicInteger res = coeff + Q - input[poly_idx * N + N_idx];
        res = output[poly_idx * N + N_idx] + modFast(res, Q);
        output[poly_idx * N + N_idx] = modFast(res, Q);
    }
}

__global__ void kernel_EvalAccCoreCGGI_mac_monic_batch(BasicInteger* output, const BasicInteger* input,
                                                 const size_t N, const BasicInteger Q, const uint32_t* indexes)
{
    const size_t poly_offset = blockIdx.x * N;
    const size_t batch_idx = blockIdx.y;
    const size_t m = indexes[batch_idx];
    const int k = m & (N - 1);
    const bool hi = (m >= N);
    for (size_t N_idx = threadIdx.x; N_idx < N; N_idx += blockDim.x)
    {
        int idx = N_idx - k;
        const bool wrapped = (idx < 0);
        if (wrapped) idx += N;

        BasicInteger coeff = input[batch_idx * 2 * N + poly_offset + idx];
        const bool neg = hi ^ wrapped;

        if (neg && coeff)
            coeff = Q - coeff;

        BasicInteger res = coeff + Q - input[batch_idx * 2 * N + poly_offset + N_idx];
        res = output[batch_idx * 2 * N + poly_offset + N_idx] + modFast(res, Q);
        output[batch_idx * 2 * N + poly_offset + N_idx] = modFast(res, Q);
    }
}