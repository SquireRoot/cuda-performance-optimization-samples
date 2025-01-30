/**
 * \file    CudaErrorHandlers.h
 * \brief   Member of the CudaTools API.
 * \author  Evan Newman
 * \date    June 2023
 */

#ifndef CUDATOOLS_CUDAHELPERS_H_
#define CUDATOOLS_CUDAHELPERS_H_

// SYSTEM INCLUDES
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <complex>

// PROJECT INCLUDES
#include <cuda.h>
#include <cufft.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cuda/std/ccomplex>

// LOCAL INCLUDES

/** Function Macro that is executed when any of the error check functions find an error
 * 
 * \param[in] msg a std::string containing the error message
*/
#define CUDATOOLS_ERROR(msg) throw std::runtime_error((msg))

/** Macro that wraps a cuda call and passes it off to the appropriate error
 *  handler along with the call string, file, and line number information of
 *  the call.
 * 
 * \param[in] call the call returning a cudaError_t, cufftResult, or cublasStatus_t
 */
#define cuErr(call) cuErrChkImpl((call), #call, __FILE__, __LINE__)

/** Handles cuda runtime errors and translate them to c++ runtime
 *  errors with verbosity. A function macro is used to get call string, 
 *  filename and line number for verbose error printing
 *
 * \param[in] result the cudaError_t result from a cuda runtime call
 * \param[in] call_str the code string that was used to call the runtime function
 * \param[in] filename the file that the call was made in
 * \param[in] line the line number that the call was made on
 */
inline void cuErrChkImpl(cudaError_t result, std::string call_str, std::string filename, int line) {
    if (result != cudaSuccess) {
        std::stringstream ss_msg;
        ss_msg << "[" << filename << ":" << line << "] CUDA error from call: " << call_str << std::endl 
              << "\t" << cudaGetErrorString(result) << std::endl;
        CUDATOOLS_ERROR(ss_msg.str());
    }
}

/** Handles cufft runtime errors and translate them to c++ runtime
 *  errors with verbosity. A function macro is used to get call string, 
 *  filename and line number for verbose error printing
 *
 * \param[in] result the cudaError_t result from a cuda runtime call
 * \param[in] call_str the code string that was used to call the runtime function
 * \param[in] filename the file that the call was made in
 * \param[in] line the line number that the call was made on
 */
inline void cuErrChkImpl(cufftResult result, std::string call_str, std::string filename, int line) {
    std::stringstream ss_msg;
    ss_msg << "[" << filename << ":" << line << "] cuFFT error from call: " << call_str << std::endl << "\tError: ";
    
    switch (result) {
        case CUFFT_SUCCESS:
            return;

        case CUFFT_INVALID_PLAN:
            ss_msg << "CUFFT_INVALID_PLAN";
            break;
        case CUFFT_ALLOC_FAILED:
            ss_msg << "CUFFT_ALLOC_FAILED";
            break;
        case CUFFT_INVALID_TYPE:
            ss_msg << "CUFFT_INVALID_TYPE";
            break;
        case CUFFT_INVALID_VALUE:
            ss_msg << "CUFFT_INVALID_VALUE";
            break;
        case CUFFT_INTERNAL_ERROR:
            ss_msg << "CUFFT_INTERNAL_ERROR";
            break;
        case CUFFT_EXEC_FAILED:
            ss_msg << "CUFFT_INTERNAL_ERROR";
            break;
        case CUFFT_SETUP_FAILED:
            ss_msg << "CUFFT_SETUP_FAILED";
            break;
        case CUFFT_INVALID_SIZE:
            ss_msg << "CUFFT_INVALID_SIZE";
            break;
        case CUFFT_UNALIGNED_DATA:
            ss_msg << "CUFFT_UNALIGNED_DATA";
            break;
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            ss_msg << "CUFFT_INCOMPLETE_PARAMETER_LIST";
            break;
        case CUFFT_INVALID_DEVICE:
            ss_msg << "CUFFT_INVALID_DEVICE";
            break;
        case CUFFT_PARSE_ERROR:
            ss_msg << "CUFFT_PARSE_ERROR";
            break;
        case CUFFT_NO_WORKSPACE:
            ss_msg << "CUFFT_NO_WORKSPACE";
            break;
        case CUFFT_NOT_IMPLEMENTED:
            ss_msg << "CUFFT_NOT_IMPLEMENTED";
            break;
        case CUFFT_LICENSE_ERROR:
            ss_msg << "CUFFT_LICENSE_ERROR";
            break;
        case CUFFT_NOT_SUPPORTED:
            ss_msg << "CUFFT_NOT_SUPPORTED";
            break;
        default:
            ss_msg << "UNKNOWN";
    }
    ss_msg << std::endl;

    CUDATOOLS_ERROR(ss_msg.str());
}

/** function to handle cuBlas runtime errors and translate them to c++ runtime
 *  errors with verbosity. A function macro is used to get call string, 
 *  filename and line number for verbose error printing
 *
 * \param[in] result the cudaError_t result from a cuda runtime call
 * \param[in] call_str the code string that was used to call the runtime function
 * \param[in] filename the file that the call was made in
 * \param[in] line the line number that the call was made on
 */
inline void cuErrChkImpl(cublasStatus_t result, std::string call_str, std::string filename, int line) {
    std::stringstream ss_msg;
    ss_msg << "[" << filename << ":" << line << "] cuBlas error from call: " << call_str << std::endl << "\tError: ";
    
    switch (result) {
        case CUBLAS_STATUS_SUCCESS:
            return;

        case CUBLAS_STATUS_ALLOC_FAILED:
            ss_msg << "CUBLAS_STATUS_ALLOC_FAILED";
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            ss_msg << "CUBLAS_STATUS_ARCH_MISMATCH";
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            ss_msg << "CUBLAS_STATUS_EXECUTION_FAILED";
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            ss_msg << "CUBLAS_STATUS_INTERNAL_ERROR";
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            ss_msg << "CUBLAS_STATUS_INVALID_VALUE";
            break;
        case CUBLAS_STATUS_LICENSE_ERROR:
            ss_msg << "CUBLAS_STATUS_LICENSE_ERROR";
            break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            ss_msg << "CUBLAS_STATUS_MAPPING_ERROR";
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            ss_msg << "CUBLAS_STATUS_NOT_INITIALIZED";
            break;
        case CUBLAS_STATUS_NOT_SUPPORTED:
            ss_msg << "CUBLAS_STATUS_NOT_SUPPORTED";
            break;
        default:
            ss_msg << "UNKNOWN";
    }
    ss_msg << std::endl;

    CUDATOOLS_ERROR(ss_msg.str());
}

#endif // #ifndef CUDATOOLS_CUDAHELPERS_H_