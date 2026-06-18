add_library(bitwise
        binfhe-base-scheme.cu
        binfhe-base-scheme-batch.cu
        binfhecontext.cu
        compact_ntt.cu
        fft.cu
        kernel.cu
        kernel-fuse.cu
        kernels_fft.cu
        rgsw-acc.cu
        rgsw-acc-batch.cu
        cirbts/ntt32_rns.cu

        arith/prng/blake2b-ref.cpp
        arith/prng/blake2xb-ref.cpp

        ntt/ntt_mont_kernels.cu
        ntt/ntt_shoup_kernels.cu)
target_link_libraries(bitwise mathdx::cufftdx)
target_include_directories(bitwise PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_definitions(bitwise PUBLIC PHANTOM_CUFFTDX_SM=${PHANTOM_CUFFTDX_SM})
target_compile_options(bitwise PUBLIC "$<$<CONFIG:Debug>:-g3>")
target_compile_options(bitwise PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
target_compile_options(bitwise PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++20>)
target_compile_options(bitwise PUBLIC $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-Xptxas=-v>)
target_compile_features(bitwise PUBLIC cxx_std_20 cuda_std_20)
