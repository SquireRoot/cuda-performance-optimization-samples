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

class BatchedExecutor {

    public:

        BatchedExecutor() = delete;
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

        size_t GetMajorBatchSize() {
            return m_work_size/m_num_streams + 1;
        }

        size_t GetMinorBatchSize() {
            return m_work_size/m_num_streams;
        }

    private:
        size_t m_work_size;
        size_t m_num_streams;

};

class CudaStream {
    public:
        CudaStream() {
            cudaStreamCreate(&m_stream);
        }
        ~CudaStream() {
            cudaStreamDestroy(m_stream);
        }

        void sync() {
            cudaStreamSynchronize(m_stream);
        }

        operator cudaStream_t() {return m_stream;}
    private:
        cudaStream_t m_stream;
};

};

#endif // #ifndef CUDATOOLS_EXECUTIONHELPERS_H