link_libraries(bitwise)

set(AES_TRANSCIPHER_COMPONENTS
    aes_transcipher/main.cu
    aes_transcipher/common.cuh
    aes_transcipher/aes128.cuh
    aes_transcipher/rlwe_tables.cuh
    aes_transcipher/gpu_lut.cuh
    aes_transcipher/extract.cuh
    aes_transcipher/key_material.cuh
    aes_gpu_ks.cuh
)

add_executable(example_aes128_transcipher ${AES_TRANSCIPHER_COMPONENTS} ${CIRBTS_CORE_SOURCES})

target_compile_features(example_aes128_transcipher PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(example_aes128_transcipher PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++20>)

if (OpenMP_CXX_FOUND)
    target_link_libraries(example_aes128_transcipher PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(example_aes128_transcipher PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif ()
