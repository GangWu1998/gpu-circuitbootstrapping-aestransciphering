#include "compact_ntt.cuh"
#include "ntt/ntt_mont.cuh"

__global__ static void
kernel_4step_ntt_forward_1024(BasicInteger* out, const BasicInteger* in,
                              const BasicInteger* tw_root_2n1_mont,
                              const BasicInteger* tw_root_2n_mont,
                              BasicInteger mod, BasicInteger mont) {
    constexpr size_t n1 = 32;
    constexpr size_t n2 = 32;
    constexpr size_t n = 1024;

    const size_t warpsPerBlock = blockDim.x / warpSize;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_root_2n1_mont[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1_mont[tid] = tw_root_2n1_mont[tid];
    }

    // transpose store into shared memory
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_forward_phase1_ws<32, 32>(s_n1n2, s_tw_root_2n1_mont,
                                               tw_root_2n_mont, mod, mont);
    __syncthreads();

    device_4step_ntt_forward_phase2_ws<32, 32>(out + blockIdx.x * n, s_n1n2, s_tw_root_2n1_mont, mod, mont);
}

__global__ static void
kernel_4step_ntt_forward_2048(BasicInteger* out, const BasicInteger* in,
                              const BasicInteger* tw_root_2n1_mont,
                              const BasicInteger* tw_root_2n_mont,
                              const BasicInteger* tw_root_n2_mont,
                              BasicInteger mod, BasicInteger mont) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 32;
    constexpr size_t n = 2048;

    const size_t warpsPerBlock = blockDim.x / warpSize;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_root_2n1_mont[n1];
    __shared__ BasicInteger s_tw_root_n2_mont[n2 / 2];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1_mont[tid] = tw_root_2n1_mont[tid];
    }

    for (size_t tid = threadIdx.x; tid < n2 / 2; tid += blockDim.x) {
        s_tw_root_n2_mont[tid] = tw_root_n2_mont[tid];
    }

    // transpose store into shared memory
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_forward_phase1_naive<64, 32>(s_n1n2, s_tw_root_2n1_mont, tw_root_2n_mont, mod, mont);
    __syncthreads();

    device_4step_ntt_forward_phase2_ws<64, 32>(out + blockIdx.x * n, s_n1n2, s_tw_root_n2_mont, mod, mont);
}

__global__ static void
kernel_4step_ntt_forward_4096(BasicInteger* out, const BasicInteger* in,
                              const BasicInteger* tw_root_2n1_mont,
                              const BasicInteger* tw_root_2n_mont,
                              BasicInteger mod, BasicInteger mont) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 64;
    constexpr size_t n = 4096;

    const size_t warpsPerBlock = blockDim.x / warpSize;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_root_2n1_mont[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1_mont[tid] = tw_root_2n1_mont[tid];
    }

    // transpose store into shared memory
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_forward_phase1_naive<64, 64>(s_n1n2, s_tw_root_2n1_mont, tw_root_2n_mont, mod, mont);
    __syncthreads();

    device_4step_ntt_forward_phase2_naive<64, 64>(out + blockIdx.x * n, s_n1n2, s_tw_root_2n1_mont, mod, mont);
}

__global__ static void
kernel_4step_ntt_inverse_1024(BasicInteger* out, const BasicInteger* in,
                              const BasicInteger* tw_inv_root_2n_mont,
                              const BasicInteger* tw_inv_root_2n1_mont,
                              BasicInteger inv_n_mont, BasicInteger mod, BasicInteger mont) {
    constexpr size_t n1 = 32;
    constexpr size_t n2 = 32;
    constexpr size_t n = 1024;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1_mont[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1_mont[tid] = tw_inv_root_2n1_mont[tid];
    }
    __syncthreads();

    device_4step_ntt_inverse_phase1_ws<32, 32>(s_n1n2, in + blockIdx.x * n,
                                               s_tw_inv_root_2n1_mont,
                                               tw_inv_root_2n_mont, mod, mont);
    __syncthreads();

    device_4step_ntt_inverse_phase2_ws<32, 32>(s_n1n2, s_tw_inv_root_2n1_mont, inv_n_mont, mod, mont);
    __syncthreads();

    // store into global memory
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            out[blockIdx.x * n + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
}

__global__ static void
kernel_4step_ntt_inverse_2048(BasicInteger* out, const BasicInteger* in,
                              const BasicInteger* tw_inv_root_n2_mont,
                              const BasicInteger* tw_inv_root_2n_mont,
                              const BasicInteger* tw_inv_root_2n1_mont,
                              BasicInteger inv_n_mont, BasicInteger mod, BasicInteger mont) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 32;
    constexpr size_t n = 2048;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1_mont[n1];
    __shared__ BasicInteger s_tw_inv_root_n2_mont[n2 / 2];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1_mont[tid] = tw_inv_root_2n1_mont[tid];
    }

    for (size_t tid = threadIdx.x; tid < n2 / 2; tid += blockDim.x) {
        s_tw_inv_root_n2_mont[tid] = tw_inv_root_n2_mont[tid];
    }
    __syncthreads();

    device_4step_ntt_inverse_phase1_ws<64, 32>(s_n1n2, in + blockIdx.x * n, s_tw_inv_root_n2_mont,
                                               tw_inv_root_2n_mont, mod, mont);
    __syncthreads();

    device_4step_ntt_inverse_phase2_naive<64, 32>(s_n1n2, s_tw_inv_root_2n1_mont,
                                                  inv_n_mont, mod, mont);
    __syncthreads();

    // store into global memory
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            out[blockIdx.x * n + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
}

__global__ static void
kernel_4step_ntt_inverse_4096(BasicInteger* out, const BasicInteger* in,
                              const BasicInteger* tw_inv_root_2n_mont,
                              const BasicInteger* tw_inv_root_2n1_mont,
                              BasicInteger inv_n_mont, BasicInteger mod, BasicInteger mont) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 64;
    constexpr size_t n = 4096;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1_mont[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1_mont[tid] = tw_inv_root_2n1_mont[tid];
    }
    __syncthreads();

    // read from global memory to shared memory
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_inverse_phase1_naive<64, 64>(s_n1n2, s_tw_inv_root_2n1_mont,
                                                  tw_inv_root_2n_mont, mod, mont);
    __syncthreads();

    device_4step_ntt_inverse_phase2_naive<64, 64>(s_n1n2, s_tw_inv_root_2n1_mont, inv_n_mont, mod, mont);
    __syncthreads();

    // store into global memory
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            out[blockIdx.x * n + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
}

void FourStepNTT::forward_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                          const cudaStream_t& stream) {
    size_t sMemSize;
    switch (n_) {
        case 1024:
            sMemSize = n1_ * (n2_ + 1) + n1_;
            kernel_4step_ntt_forward_1024<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                output, input,
                tw_root_2n1_mont_.get(), // twiddle factors for negacyclic convolution
                tw_root_2n_mont_.get(), // twiddle factors for correction step
                q_, q_inv_mod_B_);
            break;
        case 2048:
            sMemSize = n1_ * (n2_ + 1) + n1_ + n2_ / 2;
            kernel_4step_ntt_forward_2048<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                output, input,
                tw_root_2n1_mont_.get(), // twiddle factors for negacyclic convolution
                tw_root_2n_mont_.get(), // twiddle factors for correction step
                tw_root_n2_mont_.get(), // twiddle factors for cyclic convolution
                q_, q_inv_mod_B_);
            break;
        case 4096:
            sMemSize = n1_ * (n2_ + 1) + n1_;
            cudaFuncSetAttribute(kernel_4step_ntt_forward_4096, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) sMemSize);
            kernel_4step_ntt_forward_4096<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                output, input,
                tw_root_2n1_mont_.get(), // twiddle factors for negacyclic convolution
                tw_root_2n_mont_.get(), // twiddle factors for correction step
                q_, q_inv_mod_B_);
            break;
        default:
            throw std::invalid_argument("Current 4step NTT implementation requires n to be 1024 to 4096");
    }
}

void FourStepNTT::inverse_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                          const cudaStream_t& stream) {
    size_t sMemSize;
    switch (n_) {
        case 1024:
            sMemSize = n1_ * (n2_ + 1) + n1_;
            kernel_4step_ntt_inverse_1024<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                output, input,
                tw_inv_root_2n_mont_.get(), // twiddle factors for correction step
                tw_inv_root_2n1_mont_.get(), // twiddle factors for negacyclic convolution
                inv_n_mont_, q_, q_inv_mod_B_);
            break;
        case 2048:
            sMemSize = n1_ * (n2_ + 1) + n1_ + n2_ / 2;
            kernel_4step_ntt_inverse_2048<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                output, input,
                tw_inv_root_n2_mont_.get(), // twiddle factors for cyclic convolution
                tw_inv_root_2n_mont_.get(), // twiddle factors for correction step
                tw_inv_root_2n1_mont_.get(), // twiddle factors for negacyclic convolution
                inv_n_mont_, q_, q_inv_mod_B_);
            break;
        case 4096:
            sMemSize = n1_ * (n2_ + 1) + n1_;
            cudaFuncSetAttribute(kernel_4step_ntt_inverse_4096, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) sMemSize);
            kernel_4step_ntt_inverse_4096<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                output, input,
                tw_inv_root_2n_mont_.get(), // twiddle factors for correction step
                tw_inv_root_2n1_mont_.get(), // twiddle factors for negacyclic convolution
                inv_n_mont_, q_, q_inv_mod_B_);
            break;
        default:
            throw std::invalid_argument("Current 4step inverse NTT implementation requires n to be 1024 to 4096");
    }
}

__global__ void kernel_to_mont(BasicInteger* output, const BasicInteger* input, const BasicInteger R2_mod_q,
                               const BasicInteger n, const BasicInteger q, const BasicInteger q_inv_mod_R) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[blockIdx.y * n + idx] = toMont(input[blockIdx.y * n + idx], R2_mod_q, q, q_inv_mod_R);
}

void FourStepNTT::to_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                          const cudaStream_t& stream) const
{
    kernel_to_mont<<<dim3(n_ / threadsPerBlock, n_poly), threadsPerBlock, 0, stream>>>(
        output, input, B2_mod_q_, n_, q_, q_inv_mod_B_);
}

__global__ void kernel_from_mont(BasicInteger* output, const BasicInteger* input,
                                 const BasicInteger n, const BasicInteger q, const BasicInteger q_inv_mod_R) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[blockIdx.y * n + idx] = fromMont(input[blockIdx.y * n + idx], q, q_inv_mod_R);
}

void FourStepNTT::from_mont(BasicInteger* output, const BasicInteger* input, size_t threadsPerBlock, size_t n_poly,
                            const cudaStream_t& stream) const
{
    kernel_from_mont<<<dim3(n_ / threadsPerBlock, n_poly), threadsPerBlock, 0, stream>>>(
        output, input, n_, q_, q_inv_mod_B_);
}
