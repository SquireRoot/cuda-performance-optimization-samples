include(FetchContent)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(Eigen)

link_libraries(Eigen3::Eigen)
link_libraries(-lcublasLt -lcublas)
link_libraries(nvbench::main)

set(CUDA_SAMPLES_SOURCE_FILES
    MatMultBenchmarks.cu
)

add_executable(main ${CUDA_SAMPLES_SOURCE_FILES})

