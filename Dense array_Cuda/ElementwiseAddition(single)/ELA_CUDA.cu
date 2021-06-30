
/*
 * ElementwiseAddition
 * 
 * This CUDA code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0], prhs[1] := mxGPUArray or CPU Array}
 * gpuArray output, C=ELA_CUDA(A,B, alpha, beta) C=A*alpha+B*beta.
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "ElementwiseAddition.cuh"
#include "ELA_CUDA.cuh"
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DEFAULT_DIM 32   



extern "C" void  ElementwiseAddition(float * A, float * B, float * C, int numARows,
	int numAColumns,int numBRows, int numBColumns, int numCRows, int numCColumns, float alpha, float beta);
    

    
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
 int nDevices;
cudaError_t errCode =cudaGetDeviceCount(&nDevices); 
//int nDevices;
//cudaGetDeviceCount(&nDevices);

if (errCode != cudaSuccess){
printf("Error! No CUDA devices found! \n");
return;
}

    char const * const InputErrMsg = "Invalid input to MEX file, number of input arguments must be four.";
    
        if ((nrhs!=4)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }

 char *input_buf0;
 input_buf0 = mxArrayToString(prhs[0]);
 char *input_buf1;
 input_buf1 = mxArrayToString(prhs[1]);
  char *input_buf2;
 input_buf2 = mxArrayToString(prhs[2]);
 char *input_buf3;
 input_buf3 = mxArrayToString(prhs[3]);

      if ((mxIsChar(prhs[0]))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
          if ((mxIsChar(prhs[1]))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf1);
    }
      if ((mxIsChar(prhs[2]))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(THIRD ARGUMENT) must  be a scalar not  %s\n",input_buf2);
    }
          if ((mxIsChar(prhs[3]))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FOURTH ARGUMENT) must  be a scalar not  %s\n",input_buf3);
    }

if (mxIsGPUArray(prhs[0]) && mxIsGPUArray(prhs[1])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray const *B;
    mxGPUArray *C;
    float const *d_A;
    float const *d_B;
    float *d_C;
    int numARows, numAColumns, numBRows,  numBColumns, numCRows,  numCColumns;
    float alpha= mxGetScalar(prhs[2]);
    float beta = mxGetScalar(prhs[3]);


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    cudaError_t error;
    int devID = 0;
    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
     error = cudaGetDeviceProperties(&deviceProp, devID);
      if (error != cudaSuccess)
      {
          printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
          exit(EXIT_FAILURE);
      }
    int TILEDIM = (deviceProp.major < 2) ? 16 : 32;

    A = mxGPUCreateFromMxArray(prhs[0]);
    B = mxGPUCreateFromMxArray(prhs[1]);
    const mwSize *dimsA;
   dimsA=mxGPUGetDimensions(A);
   numARows = (int)dimsA[0]; /* gets number of rows of A */
   numAColumns = (int)dimsA[1]; /* gets number of columns of A */
     const mwSize *dimsB;
   dimsB=mxGPUGetDimensions(B);
   numBRows = (int)dimsB[0]; /* gets number of rows of A */
   numBColumns = (int)dimsB[1]; /* gets number of columns of A */
 		
		numCRows = numARows;

		numCColumns = numAColumns;
        
   if ((numARows != numBRows)&&(numAColumns != numBColumns)){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Array dimensions must match.%s\n");
    }
 

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be single precision (float).");
    }
    if (mxGPUGetClassID(B) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be single precision (float).");
    }

    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    d_B = (float const *)(mxGPUGetDataReadOnly(B));
    
    C = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_C = (float *)(mxGPUGetData(C));

    unsigned int TILE_DIM=16;
    dim3 dimBlock;
	dim3 dimGrid;
     switch (TILEDIM){
        
        case 16:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;

     
   MatAddSharedElementwise_GPUA<16><< <dimGrid, dimBlock >> >(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, alpha,  beta);
   
   

  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(C);
      
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
   // mxFree(dimsA);
   // mxFree(dimsB);
   
           case 32:
            
     TILE_DIM= TILEDIM;
	 dimBlock.x=TILE_DIM;
	 dimBlock.y=TILE_DIM;
     dimBlock.z=1;
	dimGrid.x = (numCColumns + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (numCRows + dimBlock.y - 1) / dimBlock.y;

     
   MatAddSharedElementwise_GPUA<32><< <dimGrid, dimBlock >> >(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, alpha,  beta);
   
   

  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device

      plhs[0] = mxGPUCreateMxArrayOnGPU(C);
      
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
   // mxFree(dimsA);
   // mxFree(dimsB);
   
   
   }
     
	}
    
    else if ((!(mxIsGPUArray(prhs[0]))) && (!(mxIsGPUArray(prhs[1])))){
 
      if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be single precision (float).");
    }
      if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(SECOND ARGUMENT) must be single precision (float).");
    }
            
            
   int numARows, numAColumns, numBRows,  numBColumns, numCRows,  numCColumns;
     numARows = (int)mxGetM(prhs[0]); 
     numAColumns = (int)mxGetN(prhs[0]);
     numBRows = (int)mxGetM(prhs[1]); 
     numBColumns = (int)mxGetN(prhs[1]);
    float alpha= mxGetScalar(prhs[2]);
    float beta = mxGetScalar(prhs[3]);
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
		numCRows = numARows;

		numCColumns = numAColumns;
    if ((numARows != numBRows)&&(numAColumns != numBColumns)){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Array dimensions must match.%s\n");
    } 
     if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be Double not  %s\n",mxGetClassID(prhs[0]));
    }
    if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be Double not  %s\n",mxGetClassID(prhs[1]));
    }
    float  * hostA ; // The A matrix
	hostA = (float *)mxGetData(prhs[0]);
    float  * hostB ; // The A matrix
	hostB = (float *)mxGetData(prhs[1]);
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    float  *pointer = (float *)mxGetPr(plhs[0]);
    //CalculateTransform(float * A, float * C, int numARows, int numAColumns, int numCRows, int numCColumns)  
    
     
       ElementwiseAddition(hostA, hostB, pointer, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, alpha,  beta);

        
        }//
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
