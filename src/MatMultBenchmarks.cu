#include <iostream>
#include <memory>
#include <limits>

#include <cuda/std/chrono>
#include <cuda_runtime.h>

#include <nvbench/nvbench.cuh>
#include "CudaTools/CudaTools.h"

#define EIGEN_NO_CUDA
#undef __CUDACC__
#include <Eigen/Core>
#define __CUDACC__

template <typename T>
class MatmulBenchEnv {
    public:
        MatmulBenchEnv(size_t N, cudaStream_t stream = cudaStreamDefault) : m_stream(stream), m_N(N) {
            // allocate host memory for a and set to all 1s
            m_A_h = std::make_shared<CudaTools::HostArray<T>>(N, N);
            m_A_h->ToEigen().setRandom();
            // std::cout << "a is:" << std::endl << a_h->ToEigen() << std::endl;

            // allocate device memory for a and transfer the host a to the device a
            m_A_d = std::make_shared<CudaTools::DeviceArray<T>>(N, N);
            m_A_h->TransferToDeviceAsync(*m_A_d, stream);

            // allocate host memory for b and set to all 1s
            m_B_h = std::make_shared<CudaTools::HostArray<T>>(N, N);
            m_B_h->ToEigen().setRandom();
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

        std::shared_ptr<CudaTools::DeviceArray<T>> GetA() { return m_A_d; }
        std::shared_ptr<CudaTools::DeviceArray<T>> GetB() { return m_B_d; }
        std::shared_ptr<CudaTools::DeviceArray<T>> GetC() { return m_C_d; }

        bool OutputIsValid() {
            m_C_d->TransferToHostAsync(*m_C_h, m_stream);
            cuErr(cudaStreamSynchronize(m_stream));

            if (!m_C_h->ToEigen().isApprox(m_C_gold_standard)) {
                return false;
            }

            return true;
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

void CuBlasMatMultBenchmark(nvbench::state& state) {
    const auto size = state.get_int64("Matrix Size");

    MatmulBenchEnv<float> env(size);

    CudaTools::CuBlasMatMult matmultop(env.GetA(), env.GetB(), env.GetC());
    matmultop();
    if (!env.OutputIsValid()) throw std::runtime_error("cublas matmultop failed");

    state.exec([&matmultop](nvbench::launch& launch) {
        matmultop(launch.get_stream());
    });
}

NVBENCH_BENCH(CuBlasMatMultBenchmark).add_int64_power_of_two_axis("Matrix Size", {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
