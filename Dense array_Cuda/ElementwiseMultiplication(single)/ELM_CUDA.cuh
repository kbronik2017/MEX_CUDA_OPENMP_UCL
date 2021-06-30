
#ifndef _ELM_CUDA_KERNEL_H_
#define _ELM_CUDA_KERNEL_H_

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
  

template <unsigned int TILE_DIM > __global__ void MatMulSharedElementwise_GPUA(float const * const A, float const * const B, float * const C,
	int numARows, int numAColumns, int numBRows, int numBColumns,
	int numCRows, int numCColumns)
{
	float CValue = 0.0;
	
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + numAColumns - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < numAColumns && Row < numARows)	As[threadIdx.y][threadIdx.x] = A[Row*numAColumns + k*TILE_DIM + threadIdx.x];
		//else	As[threadIdx.y][threadIdx.x] = 10.0;

		if (k*TILE_DIM + threadIdx.x < numBColumns && Row < numBRows)	Bs[threadIdx.y][threadIdx.x] = B[Row*numBColumns + k*TILE_DIM + threadIdx.x];
		//Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
		//else  Bs[threadIdx.y][threadIdx.x] = 50.0;

		__syncthreads();
		//for (int n = 0; n < TILE_DIM; ++n)

		CValue = As[threadIdx.y][threadIdx.x] * Bs[threadIdx.y][threadIdx.x];
		if (Row < numCRows && Col < numCColumns) C[Row*numBColumns + k*TILE_DIM + threadIdx.x] = CValue;
		__syncthreads();
	}

}

#endif // #ifndef _ELM_CUDA_KERNEL_H_
