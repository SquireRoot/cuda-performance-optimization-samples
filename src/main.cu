/**
 * \file    main.cu
 * \brief   runs all the matmult kernels sequentially for profiling purposes
 * \author  Evan Newman
 * \date    Jan 2025
 */

// SYSTEM INCLUDES
#include <iostream>

// OTHER INCLUDES
#include "CudaTools/CudaTools.h"
#include "MatMultBenchmarkEnv.h"
#include "MatMultKernels.cuh"

constexpr size_t matrix_size = 64;

int main(int argc, char** argv) {
    std::cout << "Running MatMult Kernels" << std::endl;

    Eigen::MatrixXf A_init(matrix_size, matrix_size);
    A_init.setOnes();
    // A_init.block(0, 0, matrix_size/2, matrix_size/2).setOnes();
    // A_init.setRandom();

    Eigen::MatrixXf B_init(matrix_size, matrix_size);
    B_init.setZero();
    B_init.block(0, 0, matrix_size/2, matrix_size/2).setOnes();
    B_init.block(matrix_size/2, matrix_size/2, matrix_size/2, matrix_size/2).setConstant(2);
    // B_init.setRandom();

    MatMultBenchEnv<float> env(matrix_size, A_init, B_init);

    {
        MatMult_1<float, matrix_size>(*(env.GetA()), *(env.GetB()), *(env.GetC()));

        // std::cout << env << std::endl;

        if (!env.OutputIsValid()) std::cout << "Error: MatMul_1 kernel failed" << std::endl;
        else std::cout << "MatMul_1 kernel succeeded" << std::endl;
    }

    {
        dim3 grid_size(64, 64);
        dim3 block_size(32, 32);

        MatMultBaseline<<<grid_size, block_size>>>(env.GetA()->DevicePtr(), env.GetB()->DevicePtr(), env.GetC()->DevicePtr());

        if (!env.OutputIsValid()) std::cout << "Error: MatMulBaseline kernel failed" << std::endl;
        else std::cout << "MatMulBaseline kernel succeeded" << std::endl;
    }

    return 0;
}
