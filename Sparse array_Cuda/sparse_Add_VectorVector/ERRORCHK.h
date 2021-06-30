/*
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */


#if !defined(ERRORCHK_H_)
#define ERRORCHK_H_

#include "matrix.h"
#include "mex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cusparseSafeCall(err) { __cusparseSafeCall((err), __FILE__, __LINE__); }
#define cusolverSafeCall(err) { __cusolverSafeCall((err), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, const int line)
{
	if (code != cudaSuccess)
	{
		//fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),  __FILE__, __LINE__);
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(code));
		cudaDeviceReset();
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "check the memory and process usage");
		
	}
}
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
	switch (error)
	{

	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";

	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";

	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";

	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";

	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";

	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";

	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";

	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";

	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	}

	return "<unknown>";
}

static const char *_cudasolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}
inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
	if (CUSPARSE_STATUS_SUCCESS != err) {
		//fprintf(stderr, "CUSPARSE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs", __FILE__, __LINE__, err, 
			fprintf(stderr, "CUSPARSE error in file: %s line: %d  CUSPARSE STATUS: %s   Fatal error: %s", __FILE__, __LINE__, err,
			_cusparseGetErrorEnum(err)); 

			cudaDeviceReset();
			 mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "check the fields in cusparse");
	}
}

inline void __cusolverSafeCall(cusolverStatus_t err, const char *file, const int line)
{
	if (CUSOLVER_STATUS_SUCCESS != err) {
		 
        fprintf(stderr, "CUSOLVER error in file: %s line: %d  CUSULVER STATUS: %s   Fatal error: %s", __FILE__, __LINE__, err, 
                 _cudasolverGetErrorEnum(err)); 

			cudaDeviceReset();
			mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "check the fields in cusolver");
	}
}


#endif /* !defined(ERRORCHK_H_) */
