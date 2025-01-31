/**
 * \file    Arrays.h
 * \brief   header for the HostArray and DeviceArray classes
 * \author  Evan Newman
 * \date    Nov 2023
 */

#ifndef CUDATOOLS_ARRAY_H 
#define CUDATOOLS_ARRAY_H 
 
// SYSTEM INCLUDES
#include <cuda.h>
#include <memory>

// PROJECT INCLUDES
#define EIGEN_NO_CUDA
#undef __CUDACC__
#include <Eigen/Core>
#define __CUDACC__

// LOCAL INCLUDES
#include "ErrorCheck.h"
#include "TypeConverters.h"
#include "ExecutionHelpers.h"

namespace CudaTools {

// forward declaration of the DeviceArray class
template <typename T> class DeviceArray;

/**
 * \class   HostArray
 * \brief   Host array with RAII semantics
 *
 * The HostArray manages page locked host memory
 */
template <typename T>
class HostArray
{
    public:
        using type = T;
        using EigenType = typename ToStdType<T>::type;
        using EigenMapType = Eigen::Map<Eigen::Matrix<EigenType, Eigen::Dynamic, Eigen::Dynamic>>;

    // LIFECYCLE
        // HostArray() = delete;

        /** Constructs a new cuda device array with the given dimensions 
         * 
         * \param[in]   dims    the size of each dimension in the array
        */
        HostArray(size_t inner_size, size_t outer_size = 1) : m_inner_size(inner_size), m_outer_size(outer_size) {

            cuErr(cudaMallocHost((void**) &m_data_h, sizeof(T)*m_inner_size*m_outer_size));
            memset(m_data_h, 0, sizeof(T)*m_inner_size*m_outer_size);

        }

        /** cant make copies of the object, use a smart pointer instead! I am lazy and didnt want to implement reference counting */
        HostArray(const HostArray&) = delete;

        ~HostArray() {
            cuErr(cudaFreeHost(m_data_h));
        }

    // OPERATORS

        /** cant make copies of the object, use a smart pointer instead! I am lazy and didnt want to implement reference counting */
        HostArray& operator=(const HostArray&) = delete;

        /** Array access operator
         * 
         * \param[in] idx the index to access
         * 
         * \return a reference to the value at the index
         */
        T& operator[](size_t idx) { return m_data_h[idx]; }

    // OPERATIONS
        /** Transfers the host array to a device array
         * 
         * \param[in] dest the device array to transfer data to
         */
        void TransferToDevice(DeviceArray<T>& dest) {
            if (dest.InnerSize() != m_inner_size || dest.OuterSize() != m_outer_size) {
                throw std::runtime_error("HostArray: destination array sizes do not match");
            }

            cuErr(cudaMemcpy(dest.Data(), m_data_h, sizeof(T)*m_inner_size*m_outer_size, // inner width, outer width
                               cudaMemcpyHostToDevice)); // direction, stream   
        }

        /** Transfers the host array to a device array using an asynchronous memory call
         * 
         * \param[in] dest the device array to transfer data to
         * \param[in] stream the cuda stream to queue the transfer into
         */
        void TransferToDeviceAsync(DeviceArray<T>& dest, cudaStream_t stream = cudaStreamDefault) {
            if (dest.InnerSize() != m_inner_size || dest.OuterSize() != m_outer_size) {
                throw std::runtime_error("HostArray: destination array sizes do not match");
            }

            cuErr(cudaMemcpyAsync(dest.Data(), m_data_h, sizeof(T)*m_inner_size*m_outer_size, // inner width, outer width
                                    cudaMemcpyHostToDevice, stream)); // direction, stream
        }

        /** Transfers the host array to a device array using an asynchronous memory call. Blocks of memory are transferred
         *  asynchronously across a vector of streams. Work is distributed using a BachedExecutor
         * 
         * \param[in] dest the device array to transfer data to
         * \param[in] stream the cuda streams to queue the transfer into
         */
        void TransferToDeviceAsync(DeviceArray<T>& dest, std::vector<cudaStream_t> streams) {
            if (dest.InnerSize() != m_inner_size || dest.OuterSize() != m_outer_size) {
                throw std::runtime_error("DeviceArray: destination array sizes do not match, you probably used an unsuported type");
            }

            BatchedExecutor batched_executor(dest.OuterSize(), streams.size());
            batched_executor.Execute([&] (size_t batch_size, size_t outer_idx_offset, size_t i) {
                T* dest_start = dest.Data() + outer_idx_offset*dest.InnerSize();
                T* src_start = m_data_h + outer_idx_offset*m_inner_size;

                cuErr(cudaMemcpyAsync(dest_start, src_start, sizeof(T)*m_inner_size*batch_size, // inner width, outer width
                                      cudaMemcpyHostToDevice, streams.at(i))); // direction, stream
            });
        }

        /** Set the data in this HostArray to zero
         */
        void SetZero() {
            memset(m_data_h, 0, sizeof(T)*m_inner_size*m_outer_size);
        }
 
    // ACCESS
        /** gives access to the underlying data pointer
         * 
         * \return a pointer to the backing data
        */
        T* Data() { return m_data_h; }

        /** gives access to the underlying data through an eigen map 
         * 
         * \return an eigen map of the underlying data
        */
        EigenMapType ToEigen() {
            EigenMapType returnVal((EigenType*) m_data_h, m_inner_size, m_outer_size); 
            return returnVal; 
        } 

        /** Gets the inner size of the array
         *
         * \return the inner size of the array
         */
        size_t InnerSize() { return m_inner_size; }

        /** Gets the outer size of the array
         *
         * \return the outer size of the array
         */
        size_t OuterSize() { return m_outer_size; }

    // INQUIRY
 
    private:
    
    //MEMBER VARIABLES
        const size_t m_inner_size;
        const size_t m_outer_size;

        T* m_data_h;

};

/**
 * \class   DeviceArray
 * \brief   Device array with RAII semantics
 *
 * The DeviceArray manages memory located on the GPU
 */
template <typename T>
class DeviceArray {

    public:
        using type = T;

        // DeviceArray() = delete;

        /** Constructs a new cuda device array with the given dimensions 
         * 
         * \param[in]   dims    the size of each dimension in the array
        */
        __host__ DeviceArray(size_t inner_size, size_t outer_size = 1) 
        :
            m_inner_size(inner_size), m_outer_size(outer_size) 
        {
            cuErr(cudaMalloc((void**) &m_data_d, m_outer_size*m_inner_size*sizeof(T)));
            cuErr(cudaMemset((void*) m_data_d, 0, m_outer_size*m_inner_size*sizeof(T)));

            cuErr(cudaMalloc((void**) &m_device_alloc_d, sizeof(DeviceArray<T>)));
            cuErr(cudaMemcpy(m_device_alloc_d, this, sizeof(DeviceArray<T>), cudaMemcpyHostToDevice));   
        
        }

        /** Destructs the DeviceArray
         */
        __host__ ~DeviceArray() {
            cuErr(cudaFree(m_data_d));
            cuErr(cudaFree(m_device_alloc_d));
        }
    
    // OPERATORS
        /** Array access operator
         * 
         * \param[in] idx the index to access
         * 
         * \return a reference to the value at the index
         */
        __device__ T& operator[](size_t idx) { return m_data_d[idx]; }

    // OPERATIONS

        /** Transfers the device array to the host array
         * 
         * \param[in] dest the host array to transfer data to
         */
        __host__ void TransferToHost(HostArray<T>& dest) {
            if (dest.InnerSize() != m_inner_size || dest.OuterSize() != m_outer_size) {
                throw std::runtime_error("DeviceArray: destination array sizes do not match, you probably used an unsuported type");
            }

            cuErr(cudaMemcpy(dest.Data(), m_data_d, sizeof(T)*m_inner_size*m_outer_size, cudaMemcpyDeviceToHost));
        }

        /** Transfers the device array to a host array using an asynchronous memory call
         * 
         * \param[in] dest the host array to transfer data to
         * \param[in] stream the cuda stream to queue the transfer into
         */
        __host__ void TransferToHostAsync(HostArray<T>& dest, cudaStream_t stream = cudaStreamDefault) {
            if (dest.InnerSize() != m_inner_size || dest.OuterSize() != m_outer_size) {
                throw std::runtime_error("DeviceArray: destination array sizes do not match, you probably used an unsuported type");
            }

            cuErr(cudaMemcpyAsync(dest.Data(), m_data_d, sizeof(T)*m_inner_size*m_outer_size,
                                    cudaMemcpyDeviceToHost, stream)); // direction, stream
        }


        /** Transfers the device array to a host array using an asynchronous memory call. Continuous blocks of memory are transferred
         *  asynchronously across a vector of streams. Work is distributed using a BachedExecutor
         * 
         * \param[in] dest the device array to transfer data to
         * \param[in] stream the cuda streams to queue the transfer into
         */
        __host__ void TransferToHostAsync(HostArray<T>& dest, std::vector<cudaStream_t> streams) {
            if (dest.InnerSize() != m_inner_size || dest.OuterSize() != m_outer_size) {
                throw std::runtime_error("DeviceArray: destination array sizes do not match, you probably used an unsuported type");
            }

            BatchedExecutor batched_executor(dest.OuterSize(), streams.size());
            batched_executor.Execute([&] (size_t batch_size, size_t outer_idx_offset, size_t i) {
                T* dest_start = dest.Data() + outer_idx_offset*dest.InnerSize();
                T* src_start = m_data_d + outer_idx_offset*InnerSize();

                cuErr(cudaMemcpyAsync(dest_start, src_start, sizeof(T)*m_inner_size*batch_size,
                                        cudaMemcpyDeviceToHost, streams.at(i))); // direction, stream
            });
        }

        /** Set the data in this DeviceArray to zero
         */
        __host__ void SetZero() {
            cuErr(cudaMemset((void*) m_data_d, 0, m_outer_size*m_inner_size*sizeof(T)));
        }

        /** Set the data in this DeviceArray to zero using an asynchronous memory call
         * 
         * \param[in] stream the cuda stream to queue the transfer to
         */
        __host__ void SetZeroAsync(cudaStream_t stream = cudaStreamDefault) {
            cuErr(cudaMemsetAsync((void*) m_data_d, 0, m_outer_size*m_inner_size*sizeof(T), stream));
        }

    // ACCESS
        /** gives access to the underlying data pointer
         * 
         * \return a pointer to the backing data
        */
        __host__ __device__ T* Data() { return m_data_d; };

        /** Gets the inner size of the array
         */
        __host__ __device__ size_t InnerSize() { return m_inner_size; }

        /** Gets the outer size of the array
         */
        __host__ __device__ size_t OuterSize() { return m_outer_size; }

        /** Gets a pointer to an equivalent version of this DeviceArray allocated in device memory
         */
        __host__ DeviceArray<T>* DevicePtr() { return m_device_alloc_d; }

    private:
        const size_t m_inner_size;
        const size_t m_outer_size;

        DeviceArray<T>* m_device_alloc_d;

        T* m_data_d;
};

} // namespace CudaTools

#endif // #ifndef CUDATOOLS_ARRAY_H