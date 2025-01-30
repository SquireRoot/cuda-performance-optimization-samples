/**
 * \file    CuBlasLtMatMult.h
 * \brief   header for the CuBlasLtMatMult class
 * \author  Evan Newman
 * \date    Sep 2023
 */
 
#ifndef CUDATOOLS_CUBLASLTMATMULT_H 
#define CUDATOOLS_CUBLASLTMATMULT_H
 
// SYSTEM INCLUDES
#include <memory>
#include <cuda.h>
#include <cublasLt.h>
 
// PROJECT INCLUDES
 
// LOCAL INCLUDES
#include "ErrorCheck.h"
#include "Arrays.h"
#include "TypeConverters.h"

namespace CudaTools {

/**
 * \class   CuBlasLtMatMult
 * \brief   Multiplies two matrices
 *
 * Uses cublasLt to multiply two matrices together and put the results in a third matrix
 * Matrices are defined in column major order 
 */
template <typename T>
class CuBlasLtMatMult
{   

    public:
    // LIFECYCLE

        CuBlasLtMatMult() = delete;

        /** constructs a new CuBlasLtMatMult object and completes all needed initialization
         *  this object computes C = A*B
         * 
         * \param[in] A_matrix the A device matrix to use
         * \param[in] B_matrix the B device matrix to use
         * \param[in] C_matrix the C device matrix to use
        */

        CuBlasLtMatMult(std::shared_ptr<DeviceArray<T>> A_matrix,
                std::shared_ptr<DeviceArray<T>> B_matrix,
                std::shared_ptr<DeviceArray<T>> C_matrix) :
            
            CuBlasLtMatMult(std::move(std::make_shared<T>(1.0)),
            A_matrix,
            CUBLAS_OP_N,
            B_matrix,
            CUBLAS_OP_N,
            std::move(std::make_shared<T>(0.0)),
            C_matrix,
            C_matrix) {}

        /** Constructs a new CuBlasLtMatMult object and completes all needed initialization.
         *  this object computes D = alpha*op(A)*op(B)+beta*op(C)
         * 
         * \param[in] alpha the alpha to use
         * \param[in] A_matrix the A device matrix to use
         * \param[in] A_op the operation op() that will be performed on A
         * \param[in] B_matrix the B device matrix to use
         * \param[in] B_op the operation op() that will be performed on B
         * \param[in] beta the beta to use
         * \param[in] C_matrix the C device matrix to use, C is allowed to be the same as D
         * \param[in] D_matrix the D device matrix to use, D is allowed to be the same as C
         */
        CuBlasLtMatMult(std::shared_ptr<T> alpha,
                std::shared_ptr<DeviceArray<T>> A_matrix,
                cublasOperation_t A_op,
                std::shared_ptr<DeviceArray<T>> B_matrix,
                cublasOperation_t B_op,
                std::shared_ptr<T> beta,
                std::shared_ptr<DeviceArray<T>> C_matrix,
                std::shared_ptr<DeviceArray<T>> D_matrix) 
            :
            m_alpha(alpha),
            m_A_matrix(A_matrix),
            m_B_matrix(B_matrix),
            m_beta(beta),
            m_C_matrix(C_matrix),
            m_D_matrix(D_matrix),

            m_cublas_desc{},
            m_A_layout{},
            m_B_layout{},
            m_C_layout{},
            m_D_layout{} {

            // create the cublas handle
            cuErr(cublasLtCreate(&m_cublas_handle));

            // initialize the gemm descriptor and set data and compute types
            cuErr(cublasLtMatmulDescInit(&m_cublas_desc, ToCublasType<T>::compute_type, ToCublasType<T>::data_type));

            // setting the transform of A and B,
            cuErr(cublasLtMatmulDescSetAttribute(&m_cublas_desc, CUBLASLT_MATMUL_DESC_TRANSA, &A_op, sizeof(A_op)));
            cuErr(cublasLtMatmulDescSetAttribute(&m_cublas_desc, CUBLASLT_MATMUL_DESC_TRANSB, &B_op, sizeof(B_op)));

            // initialize matrix descriptors with their respective row count, column count, and stride
            cuErr(cublasLtMatrixLayoutInit(&m_A_layout, ToCublasType<T>::data_type, A_matrix->InnerSize(), A_matrix->OuterSize(), A_matrix->InnerSize()));
            cuErr(cublasLtMatrixLayoutInit(&m_B_layout, ToCublasType<T>::data_type, B_matrix->InnerSize(), B_matrix->OuterSize(), B_matrix->InnerSize()));
            cuErr(cublasLtMatrixLayoutInit(&m_C_layout, ToCublasType<T>::data_type, C_matrix->InnerSize(), C_matrix->OuterSize(), C_matrix->InnerSize()));
            cuErr(cublasLtMatrixLayoutInit(&m_D_layout, ToCublasType<T>::data_type, D_matrix->InnerSize(), D_matrix->OuterSize(), D_matrix->InnerSize()));

            // initialize the gemm algorithm with types and the default algorithm id
            cuErr(cublasLtMatmulAlgoInit(m_cublas_handle,  // cublas handle
                                         ToCublasType<T>::compute_type,   // compute type
                                         ToCublasType<T>::data_type,   // data type of scaling factors alpha and beta
                                         ToCublasType<T>::data_type,   // data type of A
                                         ToCublasType<T>::data_type,   // data type of B
                                         ToCublasType<T>::data_type,   // data type of C
                                         ToCublasType<T>::data_type,   // data type of D
                                         CUBLAS_GEMM_DEFAULT, // use default gemm algorithm
                                         &m_cublas_algo));

            // set the inner gemm tile size
            cuErr(cublasLtMatmulAlgoConfigSetAttribute(&m_cublas_algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &m_tile_size, sizeof(m_tile_size)));

            // set the reduction mode
            cuErr(cublasLtMatmulAlgoConfigSetAttribute(&m_cublas_algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &m_reduction_scheme, sizeof(m_reduction_scheme)));

            // set the split k factor
            cuErr(cublasLtMatmulAlgoConfigSetAttribute(&m_cublas_algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &m_split_k_factor, sizeof(m_split_k_factor)));

            // check that the CuBlasLtMatMult we want to do is supported by the system
            cublasLtMatmulHeuristicResult_t result;
            cuErr(cublasLtMatmulAlgoCheck(m_cublas_handle,
                                        &m_cublas_desc,
                                        &m_A_layout,
                                        &m_B_layout,
                                        &m_C_layout,
                                        &m_D_layout,
                                        &m_cublas_algo,
                                        &result));

            m_workspace_size = 4 * 1024 * 1024;
            // m_workspace_size = result.workspaceSize;
            // std::cout << "Required workspace size is " << m_workspace_size << std::endl;

            // allocate workspace for cublaslt on gpu and initialize to 0
            cuErr(cudaMalloc((void**) &m_workspace_d, m_workspace_size)); // change this to 32 mb if on the hopper arch
            cuErr(cudaMemset(m_workspace_d, 0, m_workspace_size));

        }

        /* no copy construction, use a smart pointer if you need this */
        CuBlasLtMatMult(const CuBlasLtMatMult& toCopy) = delete;
 
        /** Destructs the CudaMatMul object */
        ~CuBlasLtMatMult(void) {
            cuErr(cublasLtDestroy(m_cublas_handle));
            cuErr(cudaFree(m_workspace_d));
        }
 
    // OPERATORS
        /* no copy assignment, use a smart pointer if you need this behavior*/
        CuBlasLtMatMult& operator=(const CuBlasLtMatMult&) = delete;

    // OPERATIONS
        void operator()(cudaStream_t stream = cudaStreamDefault) {
            // std::cout << "CudaTools cuBLASLt matmul performing multiply D = alpha*A*B+beta*C with:" << std::endl;
            // std::cout << "D: (" << m_D_matrix->InnerSize() << " x " << m_D_matrix->OuterSize() << ")" << std::endl;
            // std::cout << "alpha = " << *m_alpha << std::endl;
            // std::cout << "A: (" << m_A_matrix->InnerSize() << " x " << m_A_matrix->OuterSize() << ")" << std::endl;
            // std::cout << "B: (" << m_B_matrix->InnerSize() << " x " << m_B_matrix->OuterSize() << ")" << std::endl;
            // std::cout << "beta = " << *m_beta << std::endl;
            // std::cout << "C: (" << m_C_matrix->InnerSize() << " x " << m_C_matrix->OuterSize() << ")" << std::endl;
            // std::cout << "workspace_size = " << m_workspace_size << std::endl;

            cuErr(cublasLtMatmul(m_cublas_handle, &m_cublas_desc,
                                 m_alpha.get(),
                                 m_A_matrix->Data(), &m_A_layout,
                                 m_B_matrix->Data(), &m_B_layout,
                                 m_beta.get(),
                                 m_C_matrix->Data(), &m_C_layout,
                                 m_D_matrix->Data(), &m_D_layout,
                                 &m_cublas_algo,
                                 &m_workspace_d, m_workspace_size,
                                 stream));
        }

 
    // ACCESS

    // INQUIRY
 
    private:
        std::shared_ptr<T> m_alpha;
        std::shared_ptr<DeviceArray<T>> m_A_matrix;
        std::shared_ptr<DeviceArray<T>> m_B_matrix;
        std::shared_ptr<T> m_beta;
        std::shared_ptr<DeviceArray<T>> m_C_matrix;
        std::shared_ptr<DeviceArray<T>> m_D_matrix;

        cublasLtHandle_t m_cublas_handle;
        cublasLtMatmulDescOpaque_t m_cublas_desc;
        cublasLtMatmulAlgo_t m_cublas_algo;

        cublasLtMatrixLayoutOpaque_t m_A_layout;
        cublasLtMatrixLayoutOpaque_t m_B_layout;
        cublasLtMatrixLayoutOpaque_t m_C_layout;
        cublasLtMatrixLayoutOpaque_t m_D_layout;

        const cublasLtMatmulTile_t m_tile_size = CUBLASLT_MATMUL_TILE_16x16;
        const cublasLtReductionScheme_t m_reduction_scheme = CUBLASLT_REDUCTION_SCHEME_INPLACE;
        const int32_t m_split_k_factor = 256;

        void* m_workspace_d;
        size_t m_workspace_size;
    //MEMBER VARIABLES
        
};
 
} // namespace CudaTools

#endif  // header guard