include(FetchContent)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(Eigen)

link_libraries(Eigen3::Eigen)
link_libraries(-lcublasLt -lcublas)

set(BENCHMARK_SOURCE_FILES
    MatMultBenchmarks.cu
)

add_executable(main-benchmark ${BENCHMARK_SOURCE_FILES})
target_link_libraries(main-benchmark nvbench::main)


set(PROFILE_SOURCE_FILES
    main.cu
)

add_executable(main-test ${PROFILE_SOURCE_FILES})
target_compile_options(main-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       -g
                       >)
