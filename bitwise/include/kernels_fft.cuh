#pragma once

#include "openfhe.h"

#include "fft.cuh"

using real_t = double;
using complex_t = cuda::std::complex<real_t>;

__global__ void kernel_EvalAccCoreDM_fft(complex_t* acc, const complex_t* dct, const complex_t* RingGSWACCKey,
                                         size_t N, size_t digitsG2);

__global__ void kernel_EvalAccCoreDM_fft_batch(complex_t* acc, const complex_t* dct, complex_t** acc_keys,
                                               size_t N, size_t digitsG2);

// __global__ void kernel_EvalAccCoreDM_1024_batch_fuse(
//     BasicInteger* acc, const BasicInteger* dct, BasicInteger** acc_keys,
//     size_t N, BasicInteger mod, BasicInteger mont, size_t digitsG2,
//     const BasicInteger* tw_inv_root_2n,
//     const BasicInteger* tw_inv_root_2n_shoup,
//     const BasicInteger* tw_inv_root_2n1,
//     const BasicInteger* tw_inv_root_2n1_shoup,
//     BasicInteger inv_n, BasicInteger inv_n_shoup,
//     int IsCompositeNTT, BasicInteger P);

__global__ void kernel_EvalAccCoreCGGI_fft(complex_t* acc, const complex_t* dct, const complex_t* d_ACCKey,
                                           size_t N, size_t digitsG2);

__global__ void kernel_EvalAccCoreCGGI_fft_batch(complex_t* acc, const complex_t* dct, const complex_t* d_ACCKey,
                                                 size_t N, size_t digitsG2);

__global__ void kernel_EvalAccCoreCGGI_mac_monic(BasicInteger* output, const BasicInteger* input,
                                                 size_t N, BasicInteger Q, size_t indexPos);

__global__ void kernel_EvalAccCoreCGGI_mac_monic_batch(BasicInteger* output, const BasicInteger* input,
                                                       size_t N, BasicInteger Q, const uint32_t* indexes);

// __global__ void kernel_EvalAccCoreCGGI_batch(BasicInteger* acc, const BasicInteger* dct,
//                                              const BasicInteger* acc_key0, const BasicInteger* acc_key1,
//                                              const BasicInteger* monic_polys, size_t N,
//                                              BasicInteger mod, BasicInteger mont, size_t digitsG2,
//                                              const uint32_t* indexPos_batch, const uint32_t* indexNeg_batch);
//
// __global__ void kernel_EvalAccCoreCGGI_binary_batch(BasicInteger* acc, const BasicInteger* dct,
//                                                     const BasicInteger* acc_key,
//                                                     const BasicInteger* monic_polys, size_t N,
//                                                     BasicInteger mod, BasicInteger mont, size_t digitsG2,
//                                                     const uint32_t* indexPos_batch);

// __global__ void kernel_EvalAccCoreCGGI_1024_batch_fuse(
//     BasicInteger* acc, const BasicInteger* dct,
//     const BasicInteger* acc_key0, const BasicInteger* acc_key1,
//     const BasicInteger* monic_polys,
//     size_t N, BasicInteger mod, BasicInteger mont, size_t digitsG2,
//     const uint32_t* indexPos_batch,
//     const uint32_t* indexNeg_batch,
//     const BasicInteger* tw_inv_root_2n,
//     const BasicInteger* tw_inv_root_2n_shoup,
//     const BasicInteger* tw_inv_root_2n1,
//     const BasicInteger* tw_inv_root_2n1_shoup,
//     BasicInteger inv_n, BasicInteger inv_n_shoup,
//     int IsCompositeNTT, BasicInteger P, BasicInteger Q);
//
// __global__ void kernel_EvalAccCoreCGGI_1024_binary_batch_fuse(
//         BasicInteger *acc, const BasicInteger *dct,
//         const BasicInteger *acc_key,
//         const BasicInteger *monic_polys,
//         size_t N, BasicInteger mod, BasicInteger mont, size_t digitsG2,
//         const uint32_t *indexPos_batch,
//         const BasicInteger *tw_inv_root_2n,
//         const BasicInteger *tw_inv_root_2n_shoup,
//         const BasicInteger *tw_inv_root_2n1,
//         const BasicInteger *tw_inv_root_2n1_shoup,
//         BasicInteger inv_n, BasicInteger inv_n_shoup,
//         int IsCompositeNTT, BasicInteger P, BasicInteger Q);

// __global__ void kernel_element_add(BasicInteger* output, const BasicInteger* input1, const BasicInteger* input2,
//                                    size_t dim, BasicInteger mod);
