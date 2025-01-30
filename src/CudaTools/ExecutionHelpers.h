/**
 * \file    ExecutionHelpers.h
 * \brief   header for the ExecutionHelpers class
 * \author  Evan Newman
 * \date    Sep 2023
 */
 
#ifndef CUDATOOLS_EXECUTION_HELPERS_H 
#define CUDATOOLS_EXECUTION_HELPERS_H
 
// SYSTEM INCLUDES
#include <cuda.h>
#include <utility>
 
// PROJECT INCLUDES
 
// LOCAL INCLUDES
#include "ErrorCheck.h"
#include "Arrays.h"

namespace CudaTools {

/**
 * \class   BatchedExecutor
 * \brief   Wrapper to call a function into load balanced blocks
 *
 * Batched Executor's execute function calls the given function with load balanced block sizes
 */
class BatchedExecutor {

    public:

        /** Constructs a batched executor
         * 
         * \param[in] work_size the total number of "things" that need to be done
         * \param[in] num_streams the number of streams that we want to distribute the work across
         */
        BatchedExecutor(size_t work_size, size_t num_streams) : m_work_size(work_size), m_num_streams(num_streams) {}

        /** Executes the given function for each block of memory, number of blocks is the number of streams m_data_stream_count
         * 
         * \tparam FunctionType
         * \param function A function with the signature void(int work_size, int outer_idx_offset, int i) where work_size is the number of inner vectors to process, outer_idx_offset is the index offset of the outer dimension, and i is the current block index
         */
        template <class FunctionType>
        void Execute(FunctionType function) {
            size_t outer_idx_offset = 0;
            size_t work_size_mod = m_work_size%m_num_streams;
            size_t work_size = m_work_size/m_num_streams + 1;

            for (int i = 0; i < work_size_mod; i++) {
                function(work_size, outer_idx_offset, i);
                outer_idx_offset += work_size;
            }

            work_size--;
            for (int i = work_size_mod; i < m_num_streams; i++) {
                function(work_size, outer_idx_offset, i);
                outer_idx_offset += work_size;
            }
        }

        /** Gets the size of the larger batches
         * 
         * \return the larger batch size which is equal to the minor batch size if work_size%num_streams = 0, otherwise it is the minor batch size plus one
         */
        size_t GetMajorBatchSize() {
            return m_work_size/m_num_streams + 1;
        }

        /** Gets the minor batch size
         * 
         * \return the minor batch size which is equal to work_size/num_streams
         */
        size_t GetMinorBatchSize() {
            return m_work_size/m_num_streams;
        }

    private:
        size_t m_work_size;
        size_t m_num_streams;
};

/**
 * \class   CudaStream
 * \brief   CudaStream with RAII semantics
 *
 * creates and destroys a basic cuda stream using RAII principles
 */
class CudaStream {
    public:
        /** Constructs a CUDA stream */
        CudaStream() {
            cudaStreamCreate(&m_stream);
        }

        /** Destroys a CUDA stream */
        ~CudaStream() {
            cudaStreamDestroy(m_stream);
        }

        /** synchronizes the CUDA stream  */
        void sync() {
            cudaStreamSynchronize(m_stream);
        }

        /** converter to convert the CudaStream to a cudaStream_t */
        operator cudaStream_t() {
            return m_stream;
        }

    private:
        cudaStream_t m_stream;
};

};

#endif // #ifndef CUDATOOLS_EXECUTIONHELPERS_H