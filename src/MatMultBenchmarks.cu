/**
 * \file    MatMultBenchmarks.cu
 * \brief   Contains benchmarking code for various matrix multiply kernels
 * \author  Evan Newman
 * \date    Jan 2025
 */

// SYSTEM INCLUDES
#include <iostream>
#include <memory>
#include <limits>

// DEPENDENCIES INCLUDES
#include <cuda/std/chrono>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#define EIGEN_NO_CUDA
#undef __CUDACC__
#include <Eigen/Core>
#define __CUDACC__

// LOCAL INCLUDES 
#include "CudaTools/CudaTools.h"
#include "MatMultKernels.cuh"
#include "MatMultBenchmarkEnv.h"

constexpr size_t matrix_size = 1024;

/** Benchmark function for the CuBlas matrix multiply benchmark
 * 
 * \param[in] state the nvbench state
 */
void CuBlasMatMultBenchmark(nvbench::state& state) {
    // const auto size = state.get_int64("Matrix Size");
    const size_t size = matrix_size;

    MatMultBenchEnv<float> env(size);

    CudaTools::CuBlasMatMult matmultop(env.GetA(), env.GetB(), env.GetC());
    matmultop();
    if (!env.OutputIsValid()) throw std::runtime_error("cublas matmultop failed");

    state.exec([&matmultop](nvbench::launch& launch) {
        matmultop(launch.get_stream());
    });
}

/** nvbench registration for the CuBlasMatMultBenchmark */
NVBENCH_BENCH(CuBlasMatMultBenchmark);//.add_int64_power_of_two_axis("Matrix Size", {10});


/** Benchmark function for the baseline trivial matrix multiply kernel
 * 
 * \param[in] state the nvbench state
 */
void BaselineMatMultBenchmark(nvbench::state& state) {
    // const auto size = state.get_int64("Matrix Size");
    const size_t size = matrix_size;

    MatMultBenchEnv<float> env(size);

    dim3 grid_size(64, 64);
    dim3 block_size(32, 32);

    MatMultBaseline<<<grid_size, block_size>>>(env.GetA()->DevicePtr(), env.GetB()->DevicePtr(), env.GetC()->DevicePtr());
    if (!env.OutputIsValid()) throw std::runtime_error("MatMulBaseline kernel failed");

    state.exec([&env, &grid_size, &block_size](nvbench::launch& launch) {
        MatMultBaseline<<<grid_size, block_size, 0, launch.get_stream()>>>(env.GetA()->DevicePtr(), env.GetB()->DevicePtr(), env.GetC()->DevicePtr());
    });
}

/** nvbench registration for the CuBlasMatMultBenchmark */
NVBENCH_BENCH(BaselineMatMultBenchmark);//.add_int64_power_of_two_axis("Matrix Size", {10});

/** Benchmark function for the first optimized matrix multiply kernel using blocks read to shared memory
 * 
 * \param[in] state the nvbench state
 */
void MatMult_1_Benchmark(nvbench::state& state) {

    MatMultBenchEnv<float> env(matrix_size);
    MatMult_1<float, matrix_size>(*(env.GetA()), *(env.GetB()), *(env.GetC()));
    if (!env.OutputIsValid()) throw std::runtime_error("MatMul_1 kernel failed");

    state.exec([&env](nvbench::launch& launch) {
        MatMult_1<float, matrix_size>(*(env.GetA()), *(env.GetB()), *(env.GetC()), launch.get_stream());
    });
}

/** nvbench registration for the CuBlasMatMultBenchmark */
NVBENCH_BENCH(MatMult_1_Benchmark);

