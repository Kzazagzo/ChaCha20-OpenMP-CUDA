cmake_minimum_required(VERSION 3.22)
project("OpenMP_CUDA" LANGUAGES CUDA CXX)

# Cuda
if (UNIX)
    file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CUDA/*" "${CMAKE_CURRENT_SOURCE_DIR}/OpenMP/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/lib/*" "${CMAKE_CURRENT_SOURCE_DIR}/CPU/*")
    set(CMAKE_CUDA_STANDARD 23)
    add_executable(ChaCha20 ${CUDA_SOURCES}
            src/main.cpp
    )

    set_target_properties(ChaCha20 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # OpenMP
    file(GLOB_RECURSE OPENMP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/OpenMP/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/OpenMP/*.h")
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif ()
    target_link_libraries(ChaCha20 OpenMP::OpenMP_CXX)

    set_target_properties(ChaCha20 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
