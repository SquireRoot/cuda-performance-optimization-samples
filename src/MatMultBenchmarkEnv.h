/**
 * \file    MatMultBenchmarkEnv.h
 * \brief   Contains the environment class for managing data needed for a matrix multiply
 * \author  Evan Newman
 * \date    Jan 2025
 */

#ifndef MATMULTBENCHMARKENV_H
#define MATMULTBENCHMARKENV_H

// SYSTEM INCLUDES
#include <iostream>
#include <ostream>
#include <memory>
#include <limits>

// DEPENDENCIES INCLUDES
#define EIGEN_NO_CUDA
#undef __CUDACC__
#include <Eigen/Core>
#define __CUDACC__

// LOCAL INCLUDES 
#include "CudaTools/CudaTools.h"

/**
 * \class   MatmulBenchEnv
 * \brief   class which manages and checks the data needed for a matrix multiplication on a GPU
 *
 * The MatmuBenchEnv class allocates, initializes and checks the A, B, and C arrays needed for the matrix multiply benchmarks
 */
template <typename T>
class MatMultBenchEnv {

    using EigenMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    public:
        /** Constructs a MatmulBenchEnv object using the parameters N and stream
         * 
         * \param[in] N the dimension size of the square matrices A, B, and C
         * \param[in] stream (optional) the cuda stream to work on
         */
        MatMultBenchEnv(size_t N, cudaStream_t stream = cudaStreamDefault) 
            : MatMultBenchEnv(N, EigenMat::Random(N, N), EigenMat::Random(N, N), stream) {}

        /** Constructs a MatmulBenchEnv object using the parameters N and stream
         * 
         * \param[in] N the dimension size of the square matrices A, B, and C
         * \param[in] A_init the initial values of the A matrix
         * \param[in] B_init the initial values of the B matrix
         * \param[in] stream (optional) the cuda stream to work on
         */
        MatMultBenchEnv(size_t N, const EigenMat& A_init, const EigenMat& B_init,
                       cudaStream_t stream = cudaStreamDefault) : m_stream(stream), m_N(N) {

            // allocate host memory for a and set to all 1s
            m_A_h = std::make_shared<CudaTools::HostArray<T>>(N, N);
            m_A_h->ToEigen() = A_init;
            // std::cout << "a is:" << std::endl << a_h->ToEigen() << std::endl;

            // allocate device memory for a and transfer the host a to the device a
            m_A_d = std::make_shared<CudaTools::DeviceArray<T>>(N, N);
            m_A_h->TransferToDeviceAsync(*m_A_d, stream);

            // allocate host memory for b and set to all 1s
            m_B_h = std::make_shared<CudaTools::HostArray<T>>(N, N);
            m_B_h->ToEigen() = B_init;
            // std::cout << "b is:" << std::endl << b_h->ToEigen() << std::endl;

            // allocate device memory for b and transfer the host b to the device b
            m_B_d = std::make_shared<CudaTools::DeviceArray<T>>(N, N);
            m_B_h->TransferToDeviceAsync(*m_B_d, stream);

            // calculate a gold standard for c using Eigen
            m_C_gold_standard = m_A_h->ToEigen()*m_B_h->ToEigen();

            // allocate host and device versions of c and set to zero
            m_C_h = std::make_shared<CudaTools::HostArray<T>>(N, N);
            m_C_h->SetZero();
            m_C_d = std::make_shared<CudaTools::DeviceArray<T>>(N, N);
            m_C_d->SetZeroAsync(stream);
        }

        /** Gets the A matrix in device memory
         * 
         * \return the A matrix DeviceArray
         */
        std::shared_ptr<CudaTools::DeviceArray<T>> GetA() { return m_A_d; }

        /** Gets the B matrix in device memory
         * 
         * \return the B matrix DeviceArray
         */
        std::shared_ptr<CudaTools::DeviceArray<T>> GetB() { return m_B_d; }

        /** Gets the C matrix in device memory
         * 
         * \return the C matrix DeviceArray
         */
        std::shared_ptr<CudaTools::DeviceArray<T>> GetC() { return m_C_d; }

        /** Checks if the ouput contained in the C array is a valid result of A*B
         * 
         * \return true if the result in C is correct, false otherwise
         */
        bool OutputIsValid() {
            // transfer c back to the host
            m_C_d->TransferToHostAsync(*m_C_h, m_stream);
            cuErr(cudaStreamSynchronize(m_stream));

            // check that the values in C are correct
            if (m_C_h->ToEigen().isApprox(m_C_gold_standard)) {
                return true;
            }

            // C is not right, return false
            return false;
        }

        /** gets an eigen map to the C matrix
         * 
         * \return an Eigen map to the C matrix on the host
         */
        auto GetCHost() {
            // transfer c back to the host
            m_C_d->TransferToHostAsync(*m_C_h, m_stream);
            cuErr(cudaStreamSynchronize(m_stream));

            // return the map
            return m_C_h->ToEigen();
        }

        /** overloaded stream operator for printing etc..
         * 
         * \param[in] os the output stream to add to
         * \param[in] env the MatMultBenchEnv to print
         */
        friend std::ostream& operator<<(std::ostream& os, const MatMultBenchEnv<T>& env) {
            // transfer c back to the host
            env.m_C_d->TransferToHostAsync(*(env.m_C_h), env.m_stream);
            cuErr(cudaStreamSynchronize(env.m_stream));

            os << std::endl;
            os << "A_init: " << std::endl << env.m_A_h->ToEigen() << std::endl;
            os << "B_init: " << std::endl << env.m_B_h->ToEigen() << std::endl;
            os << "C: " << std::endl << env.m_C_h->ToEigen() << std::endl;

            return os;
        }

    private:
        cudaStream_t m_stream;

        const size_t m_N;

        std::shared_ptr<CudaTools::HostArray<T>> m_A_h;
        std::shared_ptr<CudaTools::HostArray<T>> m_B_h;
        std::shared_ptr<CudaTools::HostArray<T>> m_C_h;

        std::shared_ptr<CudaTools::DeviceArray<T>> m_A_d;
        std::shared_ptr<CudaTools::DeviceArray<T>> m_B_d;
        std::shared_ptr<CudaTools::DeviceArray<T>> m_C_d;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_C_gold_standard;
};

#endif // header guard