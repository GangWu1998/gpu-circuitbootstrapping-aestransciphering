cmake_minimum_required(VERSION 3.20)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif ()

project(GPUCircuitBootstrappingAESTransciphering LANGUAGES CXX CUDA)

if (POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif ()

option(PHANTOM_AUTO_INIT_SUBMODULES "Automatically initialize missing git submodules during configure" ON)
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX assembly" ON)

function(phantom_ensure_submodule submodule_path marker_file)
    set(marker_path "${CMAKE_SOURCE_DIR}/${marker_file}")
    if (NOT EXISTS "${marker_path}" AND PHANTOM_AUTO_INIT_SUBMODULES)
        find_package(Git QUIET)
        if (Git_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive -- "${submodule_path}"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                RESULT_VARIABLE submodule_result
                OUTPUT_VARIABLE submodule_stdout
                ERROR_VARIABLE submodule_stderr
            )
            if (NOT submodule_result EQUAL 0)
                message(FATAL_ERROR
                    "Failed to initialize submodule '${submodule_path}'.\n"
                    "${submodule_stdout}\n${submodule_stderr}")
            endif ()
        endif ()
    endif ()

    if (NOT EXISTS "${marker_path}")
        message(FATAL_ERROR
            "Missing submodule '${submodule_path}'. Run:\n"
            "  git submodule update --init --recursive")
    endif ()
endfunction()

if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif ()

set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)

phantom_ensure_submodule("thirdparty/openfhe" "thirdparty/openfhe/CMakeLists.txt")
add_subdirectory(thirdparty/openfhe)

set(_mathdx_hints
    "${CMAKE_SOURCE_DIR}/thirdparty/nvidia-mathdx"
    "/opt/nvidia/mathdx/25.06"
    "/opt/nvidia/mathdx"
)
if (DEFINED ENV{MATHDX_ROOT})
    list(PREPEND _mathdx_hints "$ENV{MATHDX_ROOT}")
endif ()
find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG HINTS ${_mathdx_hints})

add_subdirectory(src)
add_subdirectory(bitwise)
