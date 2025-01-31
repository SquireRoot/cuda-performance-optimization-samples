/**
 * \file    CudaTypeConverters.h
 * \brief   contains template conversion for types
 * \author  Evan Newman
 * \date    Sep 2023
 */
 
#ifndef CUDATOOLS_CUDA_HELPERS_H 
#define CUDATOOLS_CUDA_HELPERS_H
 
// SYSTEM INCLUDES
#include <type_traits>
#include <complex>

#include <cuda.h>
#include <cuda/std/ccomplex>
#include <cublasLt.h>
#include <cufft.h>


namespace CudaTools {   

    template <typename T>
    struct IsComplex {
        static constexpr bool value = false;
    };

    template <typename T1>
    struct IsComplex<std::complex<T1>> {
        static constexpr bool value = true;
    };

    template <typename T1>
    struct IsComplex<cuda::std::complex<T1>> {
        static constexpr bool value = true;
    };

    template <>
    struct IsComplex<cufftComplex> {
        static constexpr bool value = true;
    };

    template <>
    struct IsComplex<cufftDoubleComplex> {
        static constexpr bool value = true;
    };

    /* --- Type converters for cufft --- */
    template <typename T>
    struct ToCufftType {};
    
    template <>
    struct ToCufftType<cuda::std::complex<double>> {
        using type = cufftDoubleComplex;
    };

    template <>
    struct ToCufftType<cuda::std::complex<float>> {
        using type = cufftComplex;
    };

    /* --- Type converters for cuBlasLt --- */
    template <typename T>
    struct ToCublasType {};

    template <>
    struct ToCublasType<float> {
        static constexpr cudaDataType_t data_type = CUDA_R_32F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
        using type = float;
    };

    template <>
    struct ToCublasType<double> {
        static constexpr cudaDataType_t data_type = CUDA_R_64F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
        using type = double;
    };

    template <>
    struct ToCublasType<cuda::std::complex<float>> {
        static constexpr cudaDataType_t data_type = CUDA_C_32F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
        using type = cuComplex;
    };

    template <>
    struct ToCublasType<cuda::std::complex<double>> {
        static constexpr cudaDataType_t data_type = CUDA_C_64F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
        using type = cuDoubleComplex;
    };

    /* --- Type converters for cuda complex to std complex --- */
    template <typename T>
    struct ToStdType {
        using type = T;
    };

    template <typename T1>
    struct ToStdType<cuda::std::complex<T1>> {
        using type = std::complex<T1>;
    };
}

#endif // CUDATOOLS_CUDA_HELPERS