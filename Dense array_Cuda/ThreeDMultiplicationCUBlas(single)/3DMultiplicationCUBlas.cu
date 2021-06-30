/*
 * Three dimensional Matrix Multiplication using cublas
 * 
 * This CUDA code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0], prhs[1] := mxGPUArray or CPU Array}
 * gpuArray output, C=MM3D_CUBLAS(A,B,alpha) C=A*B*alpha.
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */
#include "matrix.h"
#include "3DMultiplicationCUBlas.cuh"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

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



void ThreeDMultiplicationCUBlas(int numARows,
	int numAColumns, int numBRows,
	int numBColumns, int numCRows, int numCColumns,
	int batch_count,
	 float **A,
	 float **B,
	float **C,
	float alpha,
	float beta){
    
    cublasHandle_t handle;
	cublasCreate(&handle);
	// Create host pointer array to device matrix storage
	float **d_A, **d_B, **d_C, **h_d_A, **h_d_B, **h_d_C;
	h_d_A = (float**)malloc(batch_count*sizeof(float*));
	h_d_B = (float**)malloc(batch_count*sizeof(float*));
	h_d_C = (float**)malloc(batch_count*sizeof(float*));



	for (int i = 0; i<batch_count; i++) {
		//cudaMalloc((void**)&h_d_A[i], dim*dim*sizeof(float));
		//cudaMalloc((void**)&h_d_B[i], dim*dim*sizeof(float));
		//cudaMalloc((void**)&h_d_C[i], dim*dim*sizeof(float));
		cudaMalloc((void**)&h_d_A[i], numARows*numAColumns*sizeof(float));
		cudaMalloc((void**)&h_d_B[i], numBRows*numBColumns*sizeof(float));
		cudaMalloc((void**)&h_d_C[i], numCRows*numCColumns*sizeof(float));
	}
	// Copy the host array of device pointers to the device
	cudaMalloc((void**)&d_A, batch_count*sizeof(float*));
	cudaMalloc((void**)&d_B, batch_count*sizeof(float*));
	cudaMalloc((void**)&d_C, batch_count*sizeof(float*));
	cudaMemcpy(d_A, h_d_A, batch_count*sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_d_B, batch_count*sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_d_C, batch_count*sizeof(float*), cudaMemcpyHostToDevice);

	for (int i = 0; i<batch_count; i++) {
		//cublasSetMatrix(dim, dim, sizeof(float), A[i], dim, h_d_A[i], dim);
		//cublasSetMatrix(dim, dim, sizeof(float), B[i], dim, h_d_B[i], dim);
		//cublasSetMatrix(dim, dim, sizeof(float), C[i], dim, h_d_C[i], dim);
		//stat = cublasSetMatrix(m, n, sizeof (*a), a, m, d_a, m); /
		//# define m 5 // number of rows
		// # define n 6 // number of columns
		cublasSetMatrix(numARows, numAColumns, sizeof(float), A[i], numARows, h_d_A[i], numARows);
		cublasSetMatrix(numBRows, numBColumns, sizeof(float), B[i], numBRows, h_d_B[i], numBRows);
		cublasSetMatrix(numCRows, numCColumns, sizeof(float), C[i], numCRows, h_d_C[i], numCRows);
	}

	cublasSgemmBatched(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		numARows, numBColumns, numAColumns,
		&alpha,
		(const float**)d_A, numARows,
		(const float**)d_B, numBRows,
		&beta,
		d_C, numCRows,
		batch_count);



	for (int i = 0; i < batch_count; i++){
		cublasGetMatrix(numARows, numBColumns, sizeof(float), h_d_C[i], numARows, C[i], numARows);

	}

	for (int i = 0; i<batch_count; i++) {
		//free(A[i]);
		//free(B[i]);
		//free(C[i]);
		cudaFree(h_d_A[i]);
		cudaFree(h_d_B[i]);
		cudaFree(h_d_C[i]);
	}

	//free(A);
	//free(B);
	//free(C);
	free(h_d_A);
	free(h_d_B);
	free(h_d_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);
    return;
    
    }







