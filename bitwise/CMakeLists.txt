set(OPENFHE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/thirdparty/openfhe)
set(OPENFHE_BINARY_DIR ${CMAKE_BINARY_DIR}/thirdparty/openfhe)
include_directories(${OPENFHE_SOURCE_DIR}/src/core/include)
include_directories(${OPENFHE_SOURCE_DIR}/src/pke/include)
include_directories(${OPENFHE_SOURCE_DIR}/src/binfhe/include)
include_directories(${OPENFHE_SOURCE_DIR}/third-party/cereal/include)
include_directories(${OPENFHE_BINARY_DIR}/src/core)
link_libraries(OPENFHEcore OPENFHEpke OPENFHEbinfhe)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_libraries(Phantom)

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(PHANTOM_CUFFTDX_SM 900 CACHE STRING "cuFFTDx SM version")
option(PHANTOM_ENABLE_CUFFTDX "Enable cuFFTDx for split FFT EvalAcc" ON)

add_subdirectory(src)

set(CIRBTS_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/cirbts-backend.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/cirbts-init.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/cirbts-bootstrap.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/cirbts-batch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/cirbts-lmkcdey.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/cirbts-fft.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/cirbts-utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cirbts/kernel.cu
)

add_subdirectory(examples)
