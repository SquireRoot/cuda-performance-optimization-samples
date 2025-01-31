/**
 * \file    MatMultKernels.cuh
 * \brief   Contains function declarations for some matrix multiply kernels I wrote
 * \author  Evan Newman
 * \date    Jan 2025
 */

#ifndef MATMULKERNELS_H
#define MATMULKERNELS_H

// SYSTEM INCLUDES
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// OTHER INCLUDES
#include "CudaTools/CudaTools.h"

/** Baseline cuda kernel for a basic matrix multiply of C=A*B. Assumes a continuous column major memory order
 * 
 * \param[in] A the device array A
 * \param[in] B the device array B
 * \param[in] C the device array C
 */
template <typename T>
__global__ void MatMultBaseline(CudaTools::DeviceArray<T>* A_arr, CudaTools::DeviceArray<T>* B_arr, CudaTools::DeviceArray<T>* C_arr) {
    // printf("Hello from GPU, size is %u\n", A->InnerSize());

    T* A = A_arr->Data();
    T* B = B_arr->Data();
    T* C = C_arr->Data();

    // grid stride loop
    for (size_t c_col = threadIdx.y + blockIdx.y*blockDim.y; c_col < B_arr->OuterSize(); c_col += blockDim.y*gridDim.y) {
        for (size_t c_row = threadIdx.x + blockIdx.x*blockDim.y; c_row < A_arr->InnerSize(); c_row += blockDim.x*gridDim.x) {                    
            T inner_product_val = 0;

            for (size_t i = 0; i < A_arr->OuterSize(); i++) {
                inner_product_val += A[c_row + i*A_arr->InnerSize()]*B[c_col*B_arr->InnerSize() + i];
            }

            C[c_row + c_col*C_arr->InnerSize()] = inner_product_val;            
        }
    }
}


/** First pass at an optimized kernel for a basic matrix multiply of C=A*B. Assumes a continuous column major memory order,
 *  matrix size greater than 1024, and A, B, C, are all square matrices of the same size. This kernel breaks up the matrices into
 *  blocks and loads the blocks into shared memory before doing the matrix multiply.
 * 
 * \param[in] A the device array A
 * \param[in] B the device array B
 * \param[in] C the device array C
 */
template <typename T, size_t N, size_t SHM_BLOCK_SIZE>
__global__ void MatMult_1_kernel(CudaTools::DeviceArray<T>* A_arr, CudaTools::DeviceArray<T>* B_arr, CudaTools::DeviceArray<T>* C_arr) {

    __shared__ T A_shm[SHM_BLOCK_SIZE*SHM_BLOCK_SIZE];
    __shared__ T B_shm[SHM_BLOCK_SIZE*SHM_BLOCK_SIZE];

    T* A = A_arr->Data() + (blockDim.x*blockIdx.x + threadIdx.x) + threadIdx.y*N;
    T* B = B_arr->Data() + threadIdx.x + (blockDim.y*blockIdx.y + threadIdx.y)*N;
    T* C = C_arr->Data() + (blockDim.x*blockIdx.x + threadIdx.x) + (blockDim.y*blockIdx.y + threadIdx.y)*N;

    T inner_product_val = 0;

    // #pragma unroll
    for (size_t block_idx = 0; block_idx < N; block_idx += SHM_BLOCK_SIZE) {

        // load the block of A into A_shm
        A_shm[SHM_BLOCK_SIZE*threadIdx.y + threadIdx.x] = *A;

        // load the block of B into B_shm
        B_shm[SHM_BLOCK_SIZE*threadIdx.y + threadIdx.x] = *B;

        // sychronize to make sure the full block has been copied to shared memory before doing the multiply
        __syncthreads();

        // multiply A_shm and B_shm and add to the inner_product val
        // #pragma unroll
        for (size_t i = 0; i < SHM_BLOCK_SIZE; i++) {
            inner_product_val += A_shm[SHM_BLOCK_SIZE*i + threadIdx.x]*B_shm[SHM_BLOCK_SIZE*threadIdx.y + i];
        }

        // sync threads again to make sure we finish multiplying before we potentially load more shared memory
        __syncthreads();

        // increment A and B
        A += blockDim.y*N;
        B += blockDim.x;

    }

    // write the inner_product_val to C
    *C = inner_product_val;
}

/** CPU wrapper function for the MatMult_1_kernel that takes care of setting launch parameters and argument checking
 * 
 * \param[in] A the device array A
 * \param[in] B the device array B
 * \param[in] C the device array C
 */
template <typename T, size_t N>
void MatMult_1(CudaTools::DeviceArray<T>& A_arr, CudaTools::DeviceArray<T>& B_arr, CudaTools::DeviceArray<T>& C_arr, cudaStream_t stream = cudaStreamDefault) {
    constexpr size_t SHM_BLOCK_SIZE = 32;

    constexpr dim3 grid_size(N/SHM_BLOCK_SIZE, N/SHM_BLOCK_SIZE);
    constexpr dim3 block_size(SHM_BLOCK_SIZE, SHM_BLOCK_SIZE);

    static_assert(N >= SHM_BLOCK_SIZE, "template parameter N must be greater than SHM_BLOCK_SIZE");
    static_assert(N%SHM_BLOCK_SIZE == 0, "template parameter N must be divisible by SHM_BLOCK_SIZE");

    if (!(A_arr.InnerSize() == N && A_arr.OuterSize() == N
           && B_arr.InnerSize() == N && B_arr.OuterSize() == N
           && C_arr.InnerSize() == N && C_arr.OuterSize() == N)) {
        throw std::runtime_error("dimension check on A, B, C, failed. All dimensions must equal N");
    }

    MatMult_1_kernel<T, N, SHM_BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(A_arr.DevicePtr(), B_arr.DevicePtr(), C_arr.DevicePtr());
}

#endif // header guard