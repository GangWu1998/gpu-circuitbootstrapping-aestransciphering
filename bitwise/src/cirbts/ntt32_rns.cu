#include "cirbts/ntt32_rns.cuh"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace phantom::cirbts::experimental {
namespace {

constexpr std::uint32_t kDefaultBlockSize = 1024;

std::uint32_t log2_exact(std::size_t n) {
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("NTT32RNSPlan requires power-of-two N");
    }
    std::uint32_t log_n = 0;
    while ((std::size_t{1} << log_n) != n) {
        ++log_n;
    }
    return log_n;
}

std::uint32_t mod_pow_u32(std::uint32_t base, std::uint64_t exp, std::uint32_t mod) {
    std::uint64_t result = 1;
    std::uint64_t cur = base;
    while (exp > 0) {
        if (exp & 1U) {
            result = (result * cur) % mod;
        }
        cur = (cur * cur) % mod;
        exp >>= 1U;
    }
    return static_cast<std::uint32_t>(result);
}

std::uint32_t mod_inv_prime_u32(std::uint32_t value, std::uint32_t mod) {
    return mod_pow_u32(value, static_cast<std::uint64_t>(mod) - 2U, mod);
}

std::vector<std::uint32_t> factor_u32(std::uint32_t value) {
    std::vector<std::uint32_t> factors;
    for (std::uint32_t p = 2; static_cast<std::uint64_t>(p) * p <= value; ++p) {
        if (value % p != 0) {
            continue;
        }
        factors.push_back(p);
        while (value % p == 0) {
            value /= p;
        }
    }
    if (value > 1) {
        factors.push_back(value);
    }
    return factors;
}

std::uint32_t primitive_root_prime_u32(std::uint32_t mod) {
    const auto factors = factor_u32(mod - 1U);
    for (std::uint32_t g = 2; g < mod; ++g) {
        bool ok = true;
        for (std::uint32_t f : factors) {
            if (mod_pow_u32(g, (mod - 1U) / f, mod) == 1U) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return g;
        }
    }
    throw std::invalid_argument("failed to find primitive root for 32-bit NTT prime");
}

std::uint32_t compute_shoup32(std::uint32_t op, std::uint32_t mod) {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(op) << 32U) / mod);
}

std::size_t reverse_bits_host(std::size_t value, std::uint32_t log_n) {
    std::size_t out = 0;
    for (std::uint32_t i = 0; i < log_n; ++i) {
        out = (out << 1U) | (value & 1U);
        value >>= 1U;
    }
    return out;
}

std::vector<std::uint32_t> powers_u32(std::uint32_t root, std::size_t n, std::uint32_t mod) {
    std::vector<std::uint32_t> out(n);
    std::uint64_t cur = 1;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<std::uint32_t>(cur);
        cur = (cur * root) % mod;
    }
    return out;
}

std::vector<std::uint32_t> bitrev_powers_u32(std::uint32_t root, std::size_t n, std::uint32_t log_n,
                                             std::uint32_t mod) {
    std::vector<std::uint32_t> out(n);
    std::uint64_t cur = 1;
    for (std::size_t i = 0; i < n; ++i) {
        out[reverse_bits_host(i, log_n)] = static_cast<std::uint32_t>(cur);
        cur = (cur * root) % mod;
    }
    return out;
}

std::vector<std::uint32_t> shoup_table_u32(const std::vector<std::uint32_t>& values, std::uint32_t mod) {
    std::vector<std::uint32_t> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), [mod](std::uint32_t v) {
        return compute_shoup32(v, mod);
    });
    return out;
}

__device__ __forceinline__ std::uint32_t bit_reverse_u32(std::uint32_t value, std::uint32_t log_n) {
    value = __brev(value);
    return value >> (32U - log_n);
}

__device__ __forceinline__ std::uint32_t add_mod_u32(std::uint32_t a, std::uint32_t b, std::uint32_t mod) {
    const std::uint32_t sum = a + b;
    return (sum >= mod || sum < a) ? sum - mod : sum;
}

__device__ __forceinline__ std::uint32_t sub_mod_u32(std::uint32_t a, std::uint32_t b, std::uint32_t mod) {
    return (a >= b) ? (a - b) : (a + mod - b);
}

__device__ __forceinline__ std::uint32_t mul_shoup_u32(std::uint32_t a, std::uint32_t b, std::uint32_t b_shoup,
                                                       std::uint32_t mod) {
    const std::uint32_t quotient = __umulhi(a, b_shoup);
    std::uint32_t res = a * b - quotient * mod;
    const std::uint32_t tmp = res - mod;
    return tmp + (tmp >> 31U) * mod;
}

__device__ __forceinline__ std::uint32_t mod_fast_u32(std::uint32_t op, std::uint32_t mod) {
    const std::uint32_t tmp = op - mod;
    return tmp + (tmp >> 31U) * mod;
}

__device__ __forceinline__ void ct_butterfly_shoup_u32(std::uint32_t& x, std::uint32_t& y,
                                                       std::uint32_t tw, std::uint32_t tw_shoup,
                                                       std::uint32_t mod) {
    const std::uint32_t mod2 = mod << 1U;
    const std::uint32_t t = mul_shoup_u32(y, tw, tw_shoup, mod);
    x = mod_fast_u32(x, mod2);
    y = x + mod2 - t;
    x += t;
}

__device__ __forceinline__ void gs_butterfly_shoup_u32(std::uint32_t& x, std::uint32_t& y,
                                                       std::uint32_t tw, std::uint32_t tw_shoup,
                                                       std::uint32_t mod) {
    const std::uint32_t mod2 = mod << 1U;
    const std::uint32_t t = x + mod2 - y;
    x += y;
    x = mod_fast_u32(x, mod2);
    y = mul_shoup_u32(t, tw, tw_shoup, mod);
}

template<std::size_t n1, std::size_t n2>
__device__ void device_4step_ntt32_forward_phase1_naive(std::uint32_t s_n1n2[n1][n2 + 1],
                                                        const std::uint32_t* s_tw_root_2n1,
                                                        const std::uint32_t* s_tw_root_2n1_shoup,
                                                        const std::uint32_t* tw_root_2n,
                                                        const std::uint32_t* tw_root_2n_shoup,
                                                        std::uint32_t mod) {
    const std::size_t warps_per_block = blockDim.x / warpSize;
    const std::size_t warp_id = threadIdx.x >> 5U;
    const std::size_t lane_id = threadIdx.x & 31U;

    for (std::size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warps_per_block) {
        for (std::size_t log_m = 0; log_m < 6; ++log_m) {
            const std::size_t log_step = 5 - log_m;
            const std::size_t w_idx = lane_id >> log_step;
            const std::size_t butt_idx = ((w_idx << log_step) + lane_id) & 0x3fU;
            ct_butterfly_shoup_u32(s_n1n2[butt_idx][n2_idx], s_n1n2[butt_idx + (1U << log_step)][n2_idx],
                                   s_tw_root_2n1[(1U << log_m) + w_idx],
                                   s_tw_root_2n1_shoup[(1U << log_m) + w_idx], mod);
        }
        s_n1n2[lane_id][n2_idx] = mul_shoup_u32(s_n1n2[lane_id][n2_idx],
                                                tw_root_2n[n2_idx * n1 + lane_id],
                                                tw_root_2n_shoup[n2_idx * n1 + lane_id], mod);
        s_n1n2[lane_id + warpSize][n2_idx] = mul_shoup_u32(s_n1n2[lane_id + warpSize][n2_idx],
                                                           tw_root_2n[n2_idx * n1 + lane_id + warpSize],
                                                           tw_root_2n_shoup[n2_idx * n1 + lane_id + warpSize], mod);
    }
}

template<std::size_t n1, std::size_t n2>
__device__ void device_4step_ntt32_forward_phase2_ws(std::uint32_t* out,
                                                     const std::uint32_t s_in[n1][n2 + 1],
                                                     const std::uint32_t* s_tw_root_n2,
                                                     const std::uint32_t* s_tw_root_n2_shoup,
                                                     std::uint32_t mod) {
    const std::size_t warps_per_block = blockDim.x / warpSize;
    const std::size_t warp_id = threadIdx.x >> 5U;
    const std::size_t lane_id = threadIdx.x & 31U;

    for (std::size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warps_per_block) {
        std::uint32_t reg = s_in[n1_idx][lane_id];
        for (std::size_t log_m = 0; log_m < 5; ++log_m) {
            const std::size_t log_step = 4 - log_m;
            const std::size_t w_idx = lane_id >> (log_step + 1U);

            const std::uint32_t reg_new = __shfl_xor_sync(0xffffffff, reg, 1U << log_step);
            const std::size_t c = (lane_id >> log_step) & 1U;
            std::uint32_t left = (1U - c) * (reg - reg_new) + reg_new;
            std::uint32_t right = c * (reg - reg_new) + reg_new;
            ct_butterfly_shoup_u32(left, right, s_tw_root_n2[w_idx], s_tw_root_n2_shoup[w_idx], mod);
            reg = (1U - c) * (left - right) + right;
        }
        std::uint32_t tmp = mod_fast_u32(reg, mod * 2U);
        out[n1_idx * n2 + lane_id] = mod_fast_u32(tmp, mod);
    }
}

template<std::size_t n1, std::size_t n2>
__device__ void device_4step_ntt32_inverse_phase1_ws(std::uint32_t s_n1n2[n1][n2 + 1],
                                                     const std::uint32_t* in,
                                                     const std::uint32_t* s_tw_inv_root_n2,
                                                     const std::uint32_t* s_tw_inv_root_n2_shoup,
                                                     const std::uint32_t* tw_inv_root_2n,
                                                     const std::uint32_t* tw_inv_root_2n_shoup,
                                                     std::uint32_t mod) {
    const std::size_t warps_per_block = blockDim.x / warpSize;
    const std::size_t warp_id = threadIdx.x >> 5U;
    const std::size_t lane_id = threadIdx.x & 31U;

    for (std::size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warps_per_block) {
        std::uint32_t reg = in[n1_idx * n2 + lane_id];
        for (int log_m = 4; log_m >= 0; --log_m) {
            const std::size_t log_step = 4 - static_cast<std::size_t>(log_m);
            const std::size_t w_idx = lane_id >> (log_step + 1U);

            const std::uint32_t reg_new = __shfl_xor_sync(0xffffffff, reg, 1U << log_step);
            const std::size_t c = (lane_id >> log_step) & 1U;
            std::uint32_t left = (1U - c) * (reg - reg_new) + reg_new;
            std::uint32_t right = c * (reg - reg_new) + reg_new;
            gs_butterfly_shoup_u32(left, right, s_tw_inv_root_n2[w_idx], s_tw_inv_root_n2_shoup[w_idx], mod);
            reg = (1U - c) * (left - right) + right;
        }
        s_n1n2[n1_idx][lane_id] = mul_shoup_u32(reg,
                                                tw_inv_root_2n[n1_idx * n2 + lane_id],
                                                tw_inv_root_2n_shoup[n1_idx * n2 + lane_id], mod);
    }
}

template<std::size_t n1, std::size_t n2>
__device__ void device_4step_ntt32_inverse_phase2_naive(std::uint32_t s_n1n2[n1][n2 + 1],
                                                        const std::uint32_t* s_tw_inv_root_2n1,
                                                        const std::uint32_t* s_tw_inv_root_2n1_shoup,
                                                        std::uint32_t inv_n, std::uint32_t inv_n_shoup,
                                                        std::uint32_t mod) {
    const std::size_t warps_per_block = blockDim.x / warpSize;
    const std::size_t warp_id = threadIdx.x >> 5U;
    const std::size_t lane_id = threadIdx.x & 31U;

    for (std::size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warps_per_block) {
        for (int log_m = 5; log_m >= 0; --log_m) {
            const std::size_t log_step = 5 - static_cast<std::size_t>(log_m);
            const std::size_t w_idx = lane_id >> log_step;
            const std::size_t butt_idx = ((w_idx << log_step) + lane_id) & 0x3fU;
            gs_butterfly_shoup_u32(s_n1n2[butt_idx][n2_idx], s_n1n2[butt_idx + (1U << log_step)][n2_idx],
                                   s_tw_inv_root_2n1[(1U << static_cast<std::size_t>(log_m)) + w_idx],
                                   s_tw_inv_root_2n1_shoup[(1U << static_cast<std::size_t>(log_m)) + w_idx], mod);
        }
        std::uint32_t tmp = mul_shoup_u32(s_n1n2[lane_id][n2_idx], inv_n, inv_n_shoup, mod);
        s_n1n2[lane_id][n2_idx] = mod_fast_u32(tmp, mod);
        tmp = mul_shoup_u32(s_n1n2[lane_id + warpSize][n2_idx], inv_n, inv_n_shoup, mod);
        s_n1n2[lane_id + warpSize][n2_idx] = mod_fast_u32(tmp, mod);
    }
}

__global__ void kernel_4step_ntt32_forward_2048(std::uint32_t* out, const std::uint32_t* in,
                                                const std::uint32_t* tw_root_2n1,
                                                const std::uint32_t* tw_root_2n1_shoup,
                                                const std::uint32_t* tw_root_2n,
                                                const std::uint32_t* tw_root_2n_shoup,
                                                const std::uint32_t* tw_root_n2,
                                                const std::uint32_t* tw_root_n2_shoup,
                                                std::uint32_t mod) {
    constexpr std::size_t n1 = 64;
    constexpr std::size_t n2 = 32;
    constexpr std::size_t n = 2048;

    const std::size_t warps_per_block = blockDim.x / warpSize;
    __shared__ std::uint32_t s_n1n2[n1][n2 + 1];
    __shared__ std::uint32_t s_tw_root_2n1[n1];
    __shared__ std::uint32_t s_tw_root_2n1_shoup[n1];
    __shared__ std::uint32_t s_tw_root_n2[n2 / 2];
    __shared__ std::uint32_t s_tw_root_n2_shoup[n2 / 2];

    for (std::size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1[tid] = tw_root_2n1[tid];
        s_tw_root_2n1_shoup[tid] = tw_root_2n1_shoup[tid];
    }
#if __CUDA_ARCH__ == 1200
    __syncthreads();
#endif
    for (std::size_t tid = threadIdx.x; tid < n2 / 2; tid += blockDim.x) {
        s_tw_root_n2[tid] = tw_root_n2[tid];
        s_tw_root_n2_shoup[tid] = tw_root_n2_shoup[tid];
    }

    const std::size_t warp_id = threadIdx.x >> 5U;
    const std::size_t lane_id = threadIdx.x & 31U;
    for (std::size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warps_per_block) {
        for (std::size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize) {
            s_n1n2[n1_idx][n2_idx] = in[static_cast<std::size_t>(blockIdx.x) * n + n1_idx * n2 + n2_idx];
        }
    }
    __syncthreads();

    device_4step_ntt32_forward_phase1_naive<n1, n2>(s_n1n2, s_tw_root_2n1, s_tw_root_2n1_shoup,
                                                    tw_root_2n, tw_root_2n_shoup, mod);
    __syncthreads();
    device_4step_ntt32_forward_phase2_ws<n1, n2>(out + static_cast<std::size_t>(blockIdx.x) * n, s_n1n2,
                                                 s_tw_root_n2, s_tw_root_n2_shoup, mod);
}

__global__ void kernel_4step_ntt32_inverse_2048(std::uint32_t* out, const std::uint32_t* in,
                                                const std::uint32_t* tw_inv_root_n2,
                                                const std::uint32_t* tw_inv_root_n2_shoup,
                                                const std::uint32_t* tw_inv_root_2n,
                                                const std::uint32_t* tw_inv_root_2n_shoup,
                                                const std::uint32_t* tw_inv_root_2n1,
                                                const std::uint32_t* tw_inv_root_2n1_shoup,
                                                std::uint32_t inv_n, std::uint32_t inv_n_shoup,
                                                std::uint32_t mod) {
    constexpr std::size_t n1 = 64;
    constexpr std::size_t n2 = 32;
    constexpr std::size_t n = 2048;

    __shared__ std::uint32_t s_n1n2[n1][n2 + 1];
    __shared__ std::uint32_t s_tw_inv_root_2n1[n1];
    __shared__ std::uint32_t s_tw_inv_root_2n1_shoup[n1];
    __shared__ std::uint32_t s_tw_inv_root_n2[n2 / 2];
    __shared__ std::uint32_t s_tw_inv_root_n2_shoup[n2 / 2];

    for (std::size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1[tid] = tw_inv_root_2n1[tid];
        s_tw_inv_root_2n1_shoup[tid] = tw_inv_root_2n1_shoup[tid];
    }
    for (std::size_t tid = threadIdx.x; tid < n2 / 2; tid += blockDim.x) {
        s_tw_inv_root_n2[tid] = tw_inv_root_n2[tid];
        s_tw_inv_root_n2_shoup[tid] = tw_inv_root_n2_shoup[tid];
    }
    __syncthreads();

    device_4step_ntt32_inverse_phase1_ws<n1, n2>(s_n1n2, in + static_cast<std::size_t>(blockIdx.x) * n,
                                                 s_tw_inv_root_n2, s_tw_inv_root_n2_shoup,
                                                 tw_inv_root_2n, tw_inv_root_2n_shoup, mod);
    __syncthreads();
    device_4step_ntt32_inverse_phase2_naive<n1, n2>(s_n1n2, s_tw_inv_root_2n1, s_tw_inv_root_2n1_shoup,
                                                    inv_n, inv_n_shoup, mod);
    __syncthreads();

    const std::size_t warps_per_block = blockDim.x / warpSize;
    const std::size_t warp_id = threadIdx.x >> 5U;
    const std::size_t lane_id = threadIdx.x & 31U;
    for (std::size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warps_per_block) {
        for (std::size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize) {
            out[static_cast<std::size_t>(blockIdx.x) * n + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
        }
    }
}

__global__ void kernel_negacyclic_ntt32_forward(std::uint32_t* out, const std::uint32_t* in, std::size_t n,
                                                std::uint32_t log_n, std::uint32_t mod,
                                                const std::uint32_t* omega_pows,
                                                const std::uint32_t* omega_pows_shoup,
                                                const std::uint32_t* psi_pows,
                                                const std::uint32_t* psi_pows_shoup) {
    extern __shared__ std::uint32_t s_poly[];
    const std::uint32_t tid = threadIdx.x;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;

    for (std::size_t i = tid; i < n; i += blockDim.x) {
        const std::uint32_t x = in[base + i] % mod;
        const std::uint32_t twisted = mul_shoup_u32(x, psi_pows[i], psi_pows_shoup[i], mod);
        s_poly[bit_reverse_u32(static_cast<std::uint32_t>(i), log_n)] = twisted;
    }
    __syncthreads();

    for (std::size_t len = 2; len <= n; len <<= 1U) {
        const std::size_t half = len >> 1U;
        const std::size_t step = n / len;
        for (std::size_t b = tid; b < n / 2; b += blockDim.x) {
            const std::size_t pos = b % half;
            const std::size_t j = (b / half) * len + pos;
            const std::size_t widx = pos * step;
            const std::uint32_t u = s_poly[j];
            const std::uint32_t v = mul_shoup_u32(s_poly[j + half], omega_pows[widx], omega_pows_shoup[widx], mod);
            s_poly[j] = add_mod_u32(u, v, mod);
            s_poly[j + half] = sub_mod_u32(u, v, mod);
        }
        __syncthreads();
    }

    for (std::size_t i = tid; i < n; i += blockDim.x) {
        out[base + i] = s_poly[i];
    }
}

__global__ void kernel_negacyclic_ntt32_inverse(std::uint32_t* out, const std::uint32_t* in, std::size_t n,
                                                std::uint32_t log_n, std::uint32_t mod,
                                                const std::uint32_t* inv_omega_pows,
                                                const std::uint32_t* inv_omega_pows_shoup,
                                                const std::uint32_t* inv_psi_inv_n_pows,
                                                const std::uint32_t* inv_psi_inv_n_pows_shoup) {
    extern __shared__ std::uint32_t s_poly[];
    const std::uint32_t tid = threadIdx.x;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;

    for (std::size_t i = tid; i < n; i += blockDim.x) {
        s_poly[bit_reverse_u32(static_cast<std::uint32_t>(i), log_n)] = in[base + i] % mod;
    }
    __syncthreads();

    for (std::size_t len = 2; len <= n; len <<= 1U) {
        const std::size_t half = len >> 1U;
        const std::size_t step = n / len;
        for (std::size_t b = tid; b < n / 2; b += blockDim.x) {
            const std::size_t pos = b % half;
            const std::size_t j = (b / half) * len + pos;
            const std::size_t widx = pos * step;
            const std::uint32_t u = s_poly[j];
            const std::uint32_t v = mul_shoup_u32(s_poly[j + half], inv_omega_pows[widx],
                                                  inv_omega_pows_shoup[widx], mod);
            s_poly[j] = add_mod_u32(u, v, mod);
            s_poly[j + half] = sub_mod_u32(u, v, mod);
        }
        __syncthreads();
    }

    for (std::size_t i = tid; i < n; i += blockDim.x) {
        out[base + i] = mul_shoup_u32(s_poly[i], inv_psi_inv_n_pows[i], inv_psi_inv_n_pows_shoup[i], mod);
    }
}

__global__ void kernel_native64_to_residue32(std::uint32_t* out, const std::uint64_t* in, std::size_t total,
                                             std::uint32_t mod) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    out[idx] = static_cast<std::uint32_t>(in[idx] % mod);
}

__global__ void kernel_shoup32_from_value(std::uint32_t* out, const std::uint32_t* in, std::size_t total,
                                          std::uint32_t mod) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    out[idx] = static_cast<std::uint32_t>((static_cast<std::uint64_t>(in[idx]) << 32U) / mod);
}

__global__ void kernel_rns32_external_product_eval(std::uint32_t* out_eval,
                                                   const std::uint32_t* dct_eval,
                                                   const std::uint32_t* key_eval,
                                                   const std::uint32_t* key_eval_shoup,
                                                   std::size_t n, std::uint32_t digits,
                                                   std::uint32_t mod) {
    const std::size_t coeff = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= n) {
        return;
    }

    std::uint32_t sum0 = 0;
    std::uint32_t sum1 = 0;
    for (std::uint32_t d = 0; d < digits; ++d) {
        const std::uint32_t digit = dct_eval[static_cast<std::size_t>(d) * n + coeff];
        const std::size_t key_base = static_cast<std::size_t>(d) * 2U * n + coeff;
        sum0 = add_mod_u32(sum0,
                           mul_shoup_u32(digit, key_eval[key_base], key_eval_shoup[key_base], mod),
                           mod);
        const std::size_t key_base1 = key_base + n;
        sum1 = add_mod_u32(sum1,
                           mul_shoup_u32(digit, key_eval[key_base1], key_eval_shoup[key_base1], mod),
                           mod);
    }
    out_eval[coeff] = sum0;
    out_eval[n + coeff] = sum1;
}

std::uint32_t add_mod_host_u32(std::uint32_t a, std::uint32_t b, std::uint32_t mod) {
    const std::uint32_t sum = a + b;
    return (sum >= mod || sum < a) ? sum - mod : sum;
}

std::uint32_t sub_mod_host_u32(std::uint32_t a, std::uint32_t b, std::uint32_t mod) {
    return (a >= b) ? (a - b) : (a + mod - b);
}

std::uint32_t mul_mod_host_u32(std::uint32_t a, std::uint32_t b, std::uint32_t mod) {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(a) * b) % mod);
}

std::vector<std::uint32_t> negacyclic_convolution_host(const std::uint32_t* a, const std::uint32_t* b,
                                                       std::size_t n, std::uint32_t mod) {
    std::vector<std::uint32_t> out(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t ai = a[i];
        if (ai == 0) {
            continue;
        }
        for (std::size_t j = 0; j < n; ++j) {
            const std::uint32_t prod = mul_mod_host_u32(ai, b[j], mod);
            const std::size_t k = i + j;
            if (k < n) {
                out[k] = add_mod_host_u32(out[k], prod, mod);
            } else {
                out[k - n] = sub_mod_host_u32(out[k - n], prod, mod);
            }
        }
    }
    return out;
}

} // namespace

std::vector<std::uint32_t> NTT32RNSPlan::default_moduli() {
    return {
        998244353U,  // 119 * 2^23 + 1
        1004535809U, // 479 * 2^21 + 1
    };
}

NTT32RNSPlan::NTT32RNSPlan(std::size_t n, const std::vector<std::uint32_t>& moduli, cudaStream_t stream)
    : n_(n), log_n_(log2_exact(n)) {
    if (n_ < 2 || n_ > 4096) {
        throw std::invalid_argument("NTT32RNSPlan currently targets N in [2, 4096]");
    }
    if (moduli.empty()) {
        throw std::invalid_argument("NTT32RNSPlan requires at least one 32-bit NTT prime");
    }

    primes_.reserve(moduli.size());
    for (std::uint32_t q : moduli) {
        if (q >= (1U << 30U)) {
            throw std::invalid_argument("NTT32RNSPlan requires q < 2^30 for 32-bit lazy butterfly safety");
        }
        if ((static_cast<std::uint64_t>(q) - 1U) % (2U * n_) != 0U) {
            throw std::invalid_argument("32-bit modulus is not 1 mod 2N");
        }

        const std::uint32_t primitive = primitive_root_prime_u32(q);
        const std::uint32_t psi = mod_pow_u32(primitive, (static_cast<std::uint64_t>(q) - 1U) / (2U * n_), q);
        if (mod_pow_u32(psi, n_, q) != q - 1U || mod_pow_u32(psi, 2U * n_, q) != 1U) {
            throw std::invalid_argument("failed to derive primitive 2N-th root for 32-bit NTT prime");
        }

        const std::uint32_t omega = static_cast<std::uint32_t>((static_cast<std::uint64_t>(psi) * psi) % q);
        const std::uint32_t inv_omega = mod_inv_prime_u32(omega, q);
        const std::uint32_t inv_psi = mod_inv_prime_u32(psi, q);
        const std::uint32_t inv_n = mod_inv_prime_u32(static_cast<std::uint32_t>(n_), q);

        auto omega_pows = powers_u32(omega, n_, q);
        auto omega_pows_shoup = shoup_table_u32(omega_pows, q);
        auto inv_omega_pows = powers_u32(inv_omega, n_, q);
        auto inv_omega_pows_shoup = shoup_table_u32(inv_omega_pows, q);
        auto psi_pows = powers_u32(psi, n_, q);
        auto psi_pows_shoup = shoup_table_u32(psi_pows, q);
        auto inv_psi_inv_n_pows = powers_u32(inv_psi, n_, q);
        for (auto& v : inv_psi_inv_n_pows) {
            v = static_cast<std::uint32_t>((static_cast<std::uint64_t>(v) * inv_n) % q);
        }
        auto inv_psi_inv_n_pows_shoup = shoup_table_u32(inv_psi_inv_n_pows, q);

        PrimeData data;
        data.q = q;
        data.psi = psi;
        data.omega = omega;
        data.inv_omega = inv_omega;
        data.inv_n = inv_n;
        data.inv_n_shoup = compute_shoup32(inv_n, q);
        data.d_omega_pows = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);
        data.d_omega_pows_shoup = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);
        data.d_inv_omega_pows = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);
        data.d_inv_omega_pows_shoup = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);
        data.d_psi_pows = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);
        data.d_psi_pows_shoup = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);
        data.d_inv_psi_inv_n_pows = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);
        data.d_inv_psi_inv_n_pows_shoup = phantom::util::make_cuda_auto_ptr<std::uint32_t>(n_, stream);

        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_omega_pows.get(), omega_pows.data(), n_ * sizeof(std::uint32_t),
                                           cudaMemcpyHostToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_omega_pows_shoup.get(), omega_pows_shoup.data(),
                                           n_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_inv_omega_pows.get(), inv_omega_pows.data(),
                                           n_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_inv_omega_pows_shoup.get(), inv_omega_pows_shoup.data(),
                                           n_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_psi_pows.get(), psi_pows.data(), n_ * sizeof(std::uint32_t),
                                           cudaMemcpyHostToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_psi_pows_shoup.get(), psi_pows_shoup.data(),
                                           n_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_inv_psi_inv_n_pows.get(), inv_psi_inv_n_pows.data(),
                                           n_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(data.d_inv_psi_inv_n_pows_shoup.get(), inv_psi_inv_n_pows_shoup.data(),
                                           n_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));

        if (n_ == 2048) {
            constexpr std::size_t n1 = 64;
            constexpr std::size_t n2 = 32;
            constexpr std::uint32_t log_n1 = 6;
            constexpr std::uint32_t log_n2 = 5;

            const std::uint32_t root_2n1 = mod_pow_u32(psi, n2, q);
            const std::uint32_t inv_root_2n1 = mod_inv_prime_u32(root_2n1, q);
            const std::uint32_t root_n2 = mod_pow_u32(psi, 2U * n1, q);
            const std::uint32_t inv_root_n2 = mod_inv_prime_u32(root_n2, q);

            auto tw_root_2n1 = bitrev_powers_u32(root_2n1, n1, log_n1, q);
            auto tw_root_2n1_shoup = shoup_table_u32(tw_root_2n1, q);
            auto tw_inv_root_2n1 = bitrev_powers_u32(inv_root_2n1, n1, log_n1, q);
            auto tw_inv_root_2n1_shoup = shoup_table_u32(tw_inv_root_2n1, q);
            auto tw_root_n2 = bitrev_powers_u32(root_n2, n2 / 2, log_n2 - 1, q);
            auto tw_root_n2_shoup = shoup_table_u32(tw_root_n2, q);
            auto tw_inv_root_n2 = bitrev_powers_u32(inv_root_n2, n2 / 2, log_n2 - 1, q);
            auto tw_inv_root_n2_shoup = shoup_table_u32(tw_inv_root_n2, q);

            std::vector<std::uint32_t> tw_root_2n(n_);
            std::vector<std::uint32_t> tw_inv_root_2n(n_);
            for (std::size_t i = 0; i < n2; ++i) {
                for (std::size_t j = 0; j < n1; ++j) {
                    const std::size_t brev_j = reverse_bits_host(j, log_n1);
                    const std::size_t exp = 2U * brev_j * i + i;
                    tw_root_2n[i * n1 + j] = mod_pow_u32(psi, exp, q);
                }
            }
            for (std::size_t i = 0; i < n1; ++i) {
                const std::size_t brev_i = reverse_bits_host(i, log_n1);
                for (std::size_t j = 0; j < n2; ++j) {
                    const std::size_t exp = 2U * brev_i * j + j;
                    tw_inv_root_2n[i * n2 + j] = mod_pow_u32(inv_psi, exp, q);
                }
            }
            auto tw_root_2n_shoup = shoup_table_u32(tw_root_2n, q);
            auto tw_inv_root_2n_shoup = shoup_table_u32(tw_inv_root_2n, q);

            auto copy_vec = [stream](phantom::util::cuda_auto_ptr<std::uint32_t>& dst,
                                     const std::vector<std::uint32_t>& src) {
                dst = phantom::util::make_cuda_auto_ptr<std::uint32_t>(src.size(), stream);
                PHANTOM_CHECK_CUDA(cudaMemcpyAsync(dst.get(), src.data(), src.size() * sizeof(std::uint32_t),
                                                   cudaMemcpyHostToDevice, stream));
            };

            copy_vec(data.d_tw_root_2n1, tw_root_2n1);
            copy_vec(data.d_tw_root_2n1_shoup, tw_root_2n1_shoup);
            copy_vec(data.d_tw_inv_root_2n1, tw_inv_root_2n1);
            copy_vec(data.d_tw_inv_root_2n1_shoup, tw_inv_root_2n1_shoup);
            copy_vec(data.d_tw_root_2n, tw_root_2n);
            copy_vec(data.d_tw_root_2n_shoup, tw_root_2n_shoup);
            copy_vec(data.d_tw_inv_root_2n, tw_inv_root_2n);
            copy_vec(data.d_tw_inv_root_2n_shoup, tw_inv_root_2n_shoup);
            copy_vec(data.d_tw_root_n2, tw_root_n2);
            copy_vec(data.d_tw_root_n2_shoup, tw_root_n2_shoup);
            copy_vec(data.d_tw_inv_root_n2, tw_inv_root_n2);
            copy_vec(data.d_tw_inv_root_n2_shoup, tw_inv_root_n2_shoup);
        }
        primes_.push_back(std::move(data));
    }
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void NTT32RNSPlan::forward_limb(std::uint32_t* out, const std::uint32_t* in, std::size_t batch, std::size_t limb,
                                cudaStream_t stream) const {
    const auto& p = primes_.at(limb);
    if (n_ == 2048 && std::getenv("NTT32_NAIVE") == nullptr) {
        kernel_4step_ntt32_forward_2048<<<static_cast<unsigned>(batch), 256, 0, stream>>>(
            out, in, p.d_tw_root_2n1.get(), p.d_tw_root_2n1_shoup.get(), p.d_tw_root_2n.get(),
            p.d_tw_root_2n_shoup.get(), p.d_tw_root_n2.get(), p.d_tw_root_n2_shoup.get(), p.q);
        PHANTOM_CHECK_CUDA_LAST();
        return;
    }
    const std::size_t shmem_bytes = n_ * sizeof(std::uint32_t);
    kernel_negacyclic_ntt32_forward<<<static_cast<unsigned>(batch), kDefaultBlockSize, shmem_bytes, stream>>>(
        out, in, n_, log_n_, p.q, p.d_omega_pows.get(), p.d_omega_pows_shoup.get(), p.d_psi_pows.get(),
        p.d_psi_pows_shoup.get());
    PHANTOM_CHECK_CUDA_LAST();
}

void NTT32RNSPlan::inverse_limb(std::uint32_t* out, const std::uint32_t* in, std::size_t batch, std::size_t limb,
                                cudaStream_t stream) const {
    const auto& p = primes_.at(limb);
    if (n_ == 2048 && std::getenv("NTT32_NAIVE") == nullptr) {
        kernel_4step_ntt32_inverse_2048<<<static_cast<unsigned>(batch), 256, 0, stream>>>(
            out, in, p.d_tw_inv_root_n2.get(), p.d_tw_inv_root_n2_shoup.get(), p.d_tw_inv_root_2n.get(),
            p.d_tw_inv_root_2n_shoup.get(), p.d_tw_inv_root_2n1.get(), p.d_tw_inv_root_2n1_shoup.get(), p.inv_n,
            p.inv_n_shoup, p.q);
        PHANTOM_CHECK_CUDA_LAST();
        return;
    }
    const std::size_t shmem_bytes = n_ * sizeof(std::uint32_t);
    kernel_negacyclic_ntt32_inverse<<<static_cast<unsigned>(batch), kDefaultBlockSize, shmem_bytes, stream>>>(
        out, in, n_, log_n_, p.q, p.d_inv_omega_pows.get(), p.d_inv_omega_pows_shoup.get(),
        p.d_inv_psi_inv_n_pows.get(), p.d_inv_psi_inv_n_pows_shoup.get());
    PHANTOM_CHECK_CUDA_LAST();
}

void NTT32RNSPlan::native_coeffs_to_rns_eval(std::uint32_t* out_eval, std::uint32_t* out_eval_shoup,
                                             const std::uint64_t* in_coeffs_mod_q, std::size_t poly_count,
                                             cudaStream_t stream) const {
    if (!out_eval || !out_eval_shoup || !in_coeffs_mod_q) {
        throw std::invalid_argument("native_coeffs_to_rns_eval received a null pointer");
    }
    const std::size_t total = poly_count * n_;
    auto d_residue = phantom::util::make_cuda_auto_ptr<std::uint32_t>(total, stream);
    constexpr std::uint32_t block_size = 256;
    const std::uint32_t grid_size = static_cast<std::uint32_t>((total + block_size - 1U) / block_size);

    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::uint32_t q = modulus(limb);
        std::uint32_t* limb_eval = out_eval + limb * total;
        std::uint32_t* limb_eval_shoup = out_eval_shoup + limb * total;

        kernel_native64_to_residue32<<<grid_size, block_size, 0, stream>>>(
            d_residue.get(), in_coeffs_mod_q, total, q);
        PHANTOM_CHECK_CUDA_LAST();
        forward_limb(limb_eval, d_residue.get(), poly_count, limb, stream);
        kernel_shoup32_from_value<<<grid_size, block_size, 0, stream>>>(
            limb_eval_shoup, limb_eval, total, q);
        PHANTOM_CHECK_CUDA_LAST();
    }
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));
}

bool NTT32RNSPlan::roundtrip_self_test(std::size_t batch, cudaStream_t stream) const {
    const std::size_t limb_stride = batch * n_;
    std::vector<std::uint32_t> h_input(limb_count() * limb_stride);
    std::vector<std::uint32_t> h_output(h_input.size());
    auto d_input = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_input.size(), stream);
    auto d_tmp = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_input.size(), stream);
    auto d_output = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_input.size(), stream);

    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::uint32_t q = modulus(limb);
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t i = 0; i < n_; ++i) {
                const std::uint64_t v = 17U * i + 131U * b + 8191U * limb + 7U;
                h_input[limb * limb_stride + b * n_ + i] = static_cast<std::uint32_t>(v % q);
            }
        }
    }

    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_input.get(), h_input.data(), h_input.size() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::size_t offset = limb * limb_stride;
        forward_limb(d_tmp.get() + offset, d_input.get() + offset, batch, limb, stream);
        inverse_limb(d_output.get() + offset, d_tmp.get() + offset, batch, limb, stream);
    }
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_output.data(), d_output.get(), h_output.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));

    return h_input == h_output;
}

bool NTT32RNSPlan::external_product_self_test(std::size_t digits, cudaStream_t stream) const {
    if (digits == 0) {
        throw std::invalid_argument("external_product_self_test requires digits > 0");
    }

    const std::size_t dct_stride = digits * n_;
    const std::size_t key_stride = digits * 2U * n_;
    const std::size_t out_stride = 2U * n_;
    std::vector<std::uint32_t> h_dct(limb_count() * dct_stride);
    std::vector<std::uint32_t> h_key(limb_count() * key_stride);
    std::vector<std::uint32_t> h_out(limb_count() * out_stride);

    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::uint32_t q = modulus(limb);
        for (std::size_t d = 0; d < digits; ++d) {
            for (std::size_t i = 0; i < n_; ++i) {
                const std::uint64_t v = 17U + 97U * d + 13U * i + 8191U * limb;
                h_dct[limb * dct_stride + d * n_ + i] = static_cast<std::uint32_t>(v % q);
            }
            for (std::size_t col = 0; col < 2; ++col) {
                for (std::size_t i = 0; i < n_; ++i) {
                    const std::uint64_t v = 11U + 131U * d + 53U * col + 29U * i + 65537U * limb;
                    h_key[limb * key_stride + (d * 2U + col) * n_ + i] = static_cast<std::uint32_t>(v % q);
                }
            }
        }
    }

    auto d_dct_coeff = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_dct.size(), stream);
    auto d_dct_eval = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_dct.size(), stream);
    auto d_key_coeff = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_key.size(), stream);
    auto d_key_eval = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_key.size(), stream);
    auto d_key_eval_shoup = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_key.size(), stream);
    auto d_out_eval = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_out.size(), stream);
    auto d_out_coeff = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_out.size(), stream);

    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_dct_coeff.get(), h_dct.data(), h_dct.size() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_key_coeff.get(), h_key.data(), h_key.size() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));

    constexpr std::uint32_t block_size = 256;
    const std::uint32_t grid_size = static_cast<std::uint32_t>((n_ + block_size - 1U) / block_size);
    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::uint32_t q = modulus(limb);
        const std::size_t dct_off = limb * dct_stride;
        const std::size_t key_off = limb * key_stride;
        const std::size_t out_off = limb * out_stride;
        forward_limb(d_dct_eval.get() + dct_off, d_dct_coeff.get() + dct_off, digits, limb, stream);
        forward_limb(d_key_eval.get() + key_off, d_key_coeff.get() + key_off, digits * 2U, limb, stream);

        const std::uint32_t key_grid =
            static_cast<std::uint32_t>((key_stride + block_size - 1U) / block_size);
        kernel_shoup32_from_value<<<key_grid, block_size, 0, stream>>>(
            d_key_eval_shoup.get() + key_off, d_key_eval.get() + key_off, key_stride, q);
        PHANTOM_CHECK_CUDA_LAST();

        kernel_rns32_external_product_eval<<<grid_size, block_size, 0, stream>>>(
            d_out_eval.get() + out_off, d_dct_eval.get() + dct_off, d_key_eval.get() + key_off,
            d_key_eval_shoup.get() + key_off, n_, static_cast<std::uint32_t>(digits), q);
        PHANTOM_CHECK_CUDA_LAST();
        inverse_limb(d_out_coeff.get() + out_off, d_out_eval.get() + out_off, 2U, limb, stream);
    }

    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(h_out.data(), d_out_coeff.get(), h_out.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));

    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::uint32_t q = modulus(limb);
        for (std::size_t col = 0; col < 2; ++col) {
            std::vector<std::uint32_t> expected(n_, 0);
            for (std::size_t d = 0; d < digits; ++d) {
                const std::uint32_t* dct_poly = h_dct.data() + limb * dct_stride + d * n_;
                const std::uint32_t* key_poly = h_key.data() + limb * key_stride + (d * 2U + col) * n_;
                auto conv = negacyclic_convolution_host(dct_poly, key_poly, n_, q);
                for (std::size_t i = 0; i < n_; ++i) {
                    expected[i] = add_mod_host_u32(expected[i], conv[i], q);
                }
            }
            const std::uint32_t* got = h_out.data() + limb * out_stride + col * n_;
            for (std::size_t i = 0; i < n_; ++i) {
                if (got[i] != expected[i]) {
                    std::cerr << "[NTT32] external-product mismatch limb=" << limb
                              << " col=" << col << " coeff=" << i
                              << " expected=" << expected[i]
                              << " got=" << got[i] << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

float NTT32RNSPlan::benchmark_roundtrip_ms(std::size_t batch, std::size_t repeat, cudaStream_t stream) const {
    const std::size_t limb_stride = batch * n_;
    auto d_input = phantom::util::make_cuda_auto_ptr<std::uint32_t>(limb_count() * limb_stride, stream);
    auto d_tmp = phantom::util::make_cuda_auto_ptr<std::uint32_t>(limb_count() * limb_stride, stream);
    auto d_output = phantom::util::make_cuda_auto_ptr<std::uint32_t>(limb_count() * limb_stride, stream);
    PHANTOM_CHECK_CUDA(cudaMemsetAsync(d_input.get(), 0x11, limb_count() * limb_stride * sizeof(std::uint32_t),
                                       stream));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    PHANTOM_CHECK_CUDA(cudaEventCreate(&start));
    PHANTOM_CHECK_CUDA(cudaEventCreate(&stop));
    PHANTOM_CHECK_CUDA(cudaEventRecord(start, stream));
    for (std::size_t r = 0; r < repeat; ++r) {
        for (std::size_t limb = 0; limb < limb_count(); ++limb) {
            const std::size_t offset = limb * limb_stride;
            forward_limb(d_tmp.get() + offset, d_input.get() + offset, batch, limb, stream);
            inverse_limb(d_output.get() + offset, d_tmp.get() + offset, batch, limb, stream);
        }
    }
    PHANTOM_CHECK_CUDA(cudaEventRecord(stop, stream));
    PHANTOM_CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    PHANTOM_CHECK_CUDA(cudaEventDestroy(start));
    PHANTOM_CHECK_CUDA(cudaEventDestroy(stop));
    return ms / static_cast<float>(repeat);
}

float NTT32RNSPlan::benchmark_external_product_eval_ms(std::size_t digits, std::size_t repeat,
                                                       cudaStream_t stream) const {
    if (digits == 0 || repeat == 0) {
        return 0.0f;
    }
    const std::size_t dct_stride = digits * n_;
    const std::size_t key_stride = digits * 2U * n_;
    const std::size_t out_stride = 2U * n_;
    std::vector<std::uint32_t> h_dct(limb_count() * dct_stride);
    std::vector<std::uint32_t> h_key(limb_count() * key_stride);
    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::uint32_t q = modulus(limb);
        for (std::size_t i = 0; i < dct_stride; ++i) {
            h_dct[limb * dct_stride + i] = static_cast<std::uint32_t>((7U + 19U * i + 4099U * limb) % q);
        }
        for (std::size_t i = 0; i < key_stride; ++i) {
            h_key[limb * key_stride + i] = static_cast<std::uint32_t>((5U + 31U * i + 65537U * limb) % q);
        }
    }

    auto d_dct_eval = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_dct.size(), stream);
    auto d_key_eval = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_key.size(), stream);
    auto d_key_eval_shoup = phantom::util::make_cuda_auto_ptr<std::uint32_t>(h_key.size(), stream);
    auto d_out_eval = phantom::util::make_cuda_auto_ptr<std::uint32_t>(limb_count() * out_stride, stream);
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_dct_eval.get(), h_dct.data(), h_dct.size() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(d_key_eval.get(), h_key.data(), h_key.size() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));

    constexpr std::uint32_t block_size = 256;
    const std::uint32_t grid_size = static_cast<std::uint32_t>((n_ + block_size - 1U) / block_size);
    for (std::size_t limb = 0; limb < limb_count(); ++limb) {
        const std::uint32_t q = modulus(limb);
        const std::size_t key_off = limb * key_stride;
        const std::uint32_t key_grid =
            static_cast<std::uint32_t>((key_stride + block_size - 1U) / block_size);
        kernel_shoup32_from_value<<<key_grid, block_size, 0, stream>>>(
            d_key_eval_shoup.get() + key_off, d_key_eval.get() + key_off, key_stride, q);
        PHANTOM_CHECK_CUDA_LAST();
    }
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    PHANTOM_CHECK_CUDA(cudaEventCreate(&start));
    PHANTOM_CHECK_CUDA(cudaEventCreate(&stop));
    PHANTOM_CHECK_CUDA(cudaEventRecord(start, stream));
    for (std::size_t r = 0; r < repeat; ++r) {
        for (std::size_t limb = 0; limb < limb_count(); ++limb) {
            const std::uint32_t q = modulus(limb);
            kernel_rns32_external_product_eval<<<grid_size, block_size, 0, stream>>>(
                d_out_eval.get() + limb * out_stride,
                d_dct_eval.get() + limb * dct_stride,
                d_key_eval.get() + limb * key_stride,
                d_key_eval_shoup.get() + limb * key_stride,
                n_, static_cast<std::uint32_t>(digits), q);
            PHANTOM_CHECK_CUDA_LAST();
        }
    }
    PHANTOM_CHECK_CUDA(cudaEventRecord(stop, stream));
    PHANTOM_CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    PHANTOM_CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    PHANTOM_CHECK_CUDA(cudaEventDestroy(start));
    PHANTOM_CHECK_CUDA(cudaEventDestroy(stop));
    return ms / static_cast<float>(repeat);
}

} // namespace phantom::cirbts::experimental
