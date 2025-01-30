/**
 * \file    CuBlasMatMult.h
 * \brief   header for the CuBlasMatMult class
 * \author  Evan Newman
 * \date    Sep 2023
 */
 
#ifndef CUDATOOLS_CUBLASMATMULT_H 
#define CUDATOOLS_CUBLASMATMULT_H
 
// SYSTEM INCLUDES
#include <memory>
#include <cuda.h>
#include <cublas_v2.h>

// PROJECT INCLUDES
 
// LOCAL INCLUDES
#include "ErrorCheck.h"
#include "Arrays.h"
#include "TypeConverters.h"

namespace CudaTools {

/**
 * \class   CuBlasMatMult
 * \brief   Multiplies two matrices
 *
 * Uses cublasLt to multiply two matrices together and put the results in a third matrix
 * Matrices are defined in column major order 
 */
template <typename T>
class CuBlasMatMult
{   

    public:
    // LIFECYCLE

        CuBlasMatMult() = delete;

        /** constructs a new CuBlasMatMult object and completes all needed initialization
         *  this object computes C = A*B
         * 
         * \param[in] A_matrix the A device matrix to use
         * \param[in] B_matrix the B device matrix to use
         * \param[in] C_matrix the C device matrix to use
        */

        CuBlasMatMult(std::shared_ptr<DeviceArray<T>> A_matrix,
                std::shared_ptr<DeviceArray<T>> B_matrix,
                std::shared_ptr<DeviceArray<T>> C_matrix) :
            
            CuBlasMatMult(std::move(std::make_shared<T>(1.0)),
            A_matrix,
            CUBLAS_OP_N,
            B_matrix,
            CUBLAS_OP_N,
            std::move(std::make_shared<T>(0.0)),
            C_matrix,
            C_matrix) {}

        /** Constructs a new CuBlasMatMult object and completes all needed initialization.
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
        CuBlasMatMult(std::shared_ptr<T> alpha,
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
            m_op_A(A_op),
            m_op_B(B_op)
        {

            cuErr(cublasCreate(&m_cublas_handle));

        }

        /* no copy construction, use a smart pointer if you need this */
        CuBlasMatMult(const CuBlasMatMult& toCopy) = delete;
 
        /** Destructs the CudaMatMul object */
        ~CuBlasMatMult(void) {
            cuErr(cublasDestroy(m_cublas_handle));
        }
 
    // OPERATORS
        /* no copy assignment, use a smart pointer if you need this behavior*/
        CuBlasMatMult& operator=(const CuBlasMatMult&) = delete;

        /** Runs the matrix multiplication on the given cuda stream
         * 
         * \param[in] stream the cuda stream to run on
         */
        void operator()(cudaStream_t stream = cudaStreamDefault) {
            // std::cout << "CudaTools cublas matmul performing multiply D = alpha*A*B+beta*C with:" << std::endl;
            // std::cout << "D: (" << m_D_matrix->InnerSize() << " x " << m_D_matrix->OuterSize() << ")" << std::endl;
            // std::cout << "alpha = " << *m_alpha << std::endl;
            // std::cout << "A: (" << m_A_matrix->InnerSize() << " x " << m_A_matrix->OuterSize() << ")" << std::endl;
            // std::cout << "B: (" << m_B_matrix->InnerSize() << " x " << m_B_matrix->OuterSize() << ")" << std::endl;
            // std::cout << "beta = " << *m_beta << std::endl;
            // std::cout << "C: (" << m_C_matrix->InnerSize() << " x " << m_C_matrix->OuterSize() << ")" << std::endl;

            cuErr(cublasSetStream(m_cublas_handle, stream));

            if constexpr (std::is_same_v<typename ToCublasType<T>::type, float>) {
                cuErr(cublasSgemm(m_cublas_handle, m_op_A, m_op_B,
                                  m_A_matrix->InnerSize(), m_B_matrix->OuterSize(), m_A_matrix->OuterSize(),
                                  m_alpha.get(),
                                  m_A_matrix->Data(), m_A_matrix->InnerSize(),
                                  m_B_matrix->Data(), m_B_matrix->InnerSize(),
                                  m_beta.get(),
                                  m_C_matrix->Data(), m_C_matrix->InnerSize()));
            }
            
            if constexpr (std::is_same_v<typename ToCublasType<T>::type, double>) {
                cuErr(cublasDgemm(m_cublas_handle, m_op_A, m_op_B,
                                  m_A_matrix->InnerSize(), m_B_matrix->OuterSize(), m_A_matrix->OuterSize(),
                                  m_alpha.get(),
                                  m_A_matrix->Data(), m_A_matrix->InnerSize(),
                                  m_B_matrix->Data(), m_B_matrix->InnerSize(),
                                  m_beta.get(),
                                  m_C_matrix->Data(), m_C_matrix->InnerSize()));
            }

            if constexpr (std::is_same_v<typename ToCublasType<T>::type, cuComplex>) {
                cuErr(cublasCgemm(m_cublas_handle, m_op_A, m_op_B,
                                  m_A_matrix->InnerSize(), m_B_matrix->OuterSize(), m_A_matrix->OuterSize(),
                                  m_alpha.get(),
                                  (cuComplex*) m_A_matrix->Data(), m_A_matrix->InnerSize(),
                                  (cuComplex*) m_B_matrix->Data(), m_B_matrix->InnerSize(),
                                  m_beta.get(),
                                  (cuComplex*) m_C_matrix->Data(), m_C_matrix->InnerSize()));
            }

            if constexpr (std::is_same_v<typename ToCublasType<T>::type, cuDoubleComplex>) {
                cuErr(cublasZgemm(m_cublas_handle, m_op_A, m_op_B,
                                  m_A_matrix->InnerSize(), m_B_matrix->OuterSize(), m_A_matrix->OuterSize(),
                                  m_alpha.get(),
                                  (cuDoubleComplex*) m_A_matrix->Data(), m_A_matrix->InnerSize(),
                                  (cuDoubleComplex*) m_B_matrix->Data(), m_B_matrix->InnerSize(),
                                  m_beta.get(),
                                  (cuDoubleComplex*) m_C_matrix->Data(), m_C_matrix->InnerSize()));
            }
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

        cublasOperation_t m_op_A;
        cublasOperation_t m_op_B;

        cublasHandle_t m_cublas_handle;
    //MEMBER VARIABLES
        
};
 
} // namespace CudaTools

#endif  // header guard