# Link OpenFHE
set(OPENFHE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/thirdparty/openfhe)
set(OPENFHE_BINARY_DIR ${CMAKE_BINARY_DIR}/thirdparty/openfhe)
include_directories(${OPENFHE_SOURCE_DIR}/src/core/include)
include_directories(${OPENFHE_SOURCE_DIR}/src/pke/include)
include_directories(${OPENFHE_SOURCE_DIR}/src/binfhe/include)
include_directories(${OPENFHE_SOURCE_DIR}/third-party/cereal/include)
include_directories(${OPENFHE_BINARY_DIR}/src/core)
link_libraries(OPENFHEcore OPENFHEpke OPENFHEbinfhe)

# OpenMP (only CXX; CUDA OpenMP is not required here)
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_definitions(-DPARALLEL)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

add_library(Phantom SHARED
        batchencoder.cu
        ciphertext.cu
        ckks.cu
        context.cu
        prng.cu
        polymath.cu
        secretkey.cu
        rns.cu
        rns_base.cu
        rns_bconv.cu
        scalingvariant.cu
        evaluate.cu
        fft.cu
        galois.cu
        eval_key_switch.cu

        ntt/fntt_2d.cu
        ntt/ntt_1d.cu
        ntt/ntt_keyswitch_old.cu
        ntt/intt_2d.cu
        ntt/ntt_modup.cu
        ntt/ntt_moddown.cu

        host/blake2b.cu
        host/blake2xb.cu
        host/globals.cu
        host/hash.cu
        host/modulus.cu
        host/ntt.cu
        host/numth.cu
        host/rns.cu
        host/uintarith.cu
        host/uintarithmod.cu
        host/uintarithsmallmod.cu
)
target_include_directories(Phantom
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_options(Phantom PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
target_compile_options(Phantom PUBLIC $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>)
target_compile_options(Phantom PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=c++17>)
