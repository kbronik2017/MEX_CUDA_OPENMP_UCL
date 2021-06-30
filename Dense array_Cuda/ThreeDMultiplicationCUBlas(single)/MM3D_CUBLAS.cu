
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
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "3DMultiplicationCUBlas.cuh"
#include "MM3D_CUBLAS.cuh"
//#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>


extern "C" void ThreeDMultiplicationCUBlas(int numARows,
	int numAColumns, int numBRows,
	int numBColumns, int numCRows, int numCColumns,
	int batch_count,
	 float **A,
	 float **B,
	float **C,
	float alpha,
	float beta);
    
    
extern "C" void ThreeDMultiplicationCUBlasNGA(int numARows,
	int numAColumns, int numBRows,
	int numBColumns, int numCRows, int numCColumns,
	int batch_count,
	 float **A,
	 float **B,
	float **C,
	float alpha,
	float beta);
 
    
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

    char const * const InputErrMsg = "Invalid input to MEX file, number of input arguments must be three.";
    
        if ((nrhs!=3)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }

 char *input_buf0;
 input_buf0 = mxArrayToString(prhs[0]);
 char *input_buf1;
 input_buf1 = mxArrayToString(prhs[1]);
  char *input_buf2;
 input_buf2 = mxArrayToString(prhs[2]);
 //char *input_buf3;
 //input_buf3 = mxArrayToString(prhs[3]);

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
         // if ((mxIsChar(prhs[3]))){
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
            //    "Input(FOURTH ARGUMENT) must  be a scalar not  %s\n",input_buf3);
   // }

if (mxIsGPUArray(prhs[0]) && mxIsGPUArray(prhs[1])) {
		//mexErrMsgIdAndTxt(errId, errMsg);
           /* Declare all variables.*/
    mxGPUArray  *A;
    mxGPUArray  *B;
    mxGPUArray *C;
    float  *d_A;
    float  *d_B;
    float *d_C;
    int numARows, numAColumns, numBRows,  numBColumns, numCRows,  numCColumns;
    float alpha= mxGetScalar(prhs[2]);
    float beta = 0.0;


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();


    A = mxGPUCopyFromMxArray(prhs[0]);
    B = mxGPUCopyFromMxArray(prhs[1]);
    const mwSize *dimsA;
   dimsA=mxGPUGetDimensions(A);
   numARows = (int)dimsA[0]; /* gets number of rows of A */
   numAColumns = (int)dimsA[1]; /* gets number of columns of A */
   int batch_count = (int)dimsA[2];
     const mwSize *dimsB;
   dimsB=mxGPUGetDimensions(B);
   numBRows = (int)dimsB[0]; /* gets number of rows of A */
   numBColumns = (int)dimsB[1]; /* gets number of columns of A */
 		int batch_countB = (int)dimsB[2];
     numCRows = numARows;
	 numCColumns = numBColumns;
      
        
   if ((batch_count != batch_countB)||(numAColumns != numBRows)){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Array dimensions must match, the column number of first argument must be equal to the row number of second argument and the number of matrices must match.\n");
    }
 

    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be float\n");
    }
    if (mxGPUGetClassID(B) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be float\n");
    }

    d_A = (float*) mxGPUGetData(A);
    d_B = (float*) mxGPUGetData(B);
    
    ////////////////////
    	 float **deviceA, **deviceB, **deviceC;
	deviceA = (float**)malloc(batch_count*sizeof(float*));
	deviceB = (float**)malloc(batch_count*sizeof(float*));
	deviceC = (float**)malloc(batch_count*sizeof(float*));
    //	for (int i = 0; i<batch_count; i++) {

	//	deviceA[i] = (float*)malloc(numARows*numAColumns*sizeof(float));
	//	deviceB[i] = (float*)malloc(numBRows*numBColumns*sizeof(float));
	//	deviceC[i] = (float*)malloc(numCRows*numCColumns*sizeof(float));
//	}
    
//const mwSize * dimsC;
//mwSize  dimsC[2] = {dimsA[0],dimsB[1]};
    //mwSize  dimsC[2];
    //dimsC[0]=(int)mxGPUGetDimensions(A)[0];
    //
    //dimsC[1]=(int)mxGPUGetDimensions(B)[1];
    //////////////// OK!
     size_t pivot_dimensions[3] = {numCRows,numCColumns,batch_count};
    
    
    C = mxGPUCreateGPUArray(3,
                            (mwSize*) pivot_dimensions,
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
       //C = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                      //      mxGPUGetDimensions(A),
                       //     mxGPUGetClassID(A),
                        //    mxGPUGetComplexity(A),
                       //     MX_GPU_DO_NOT_INITIALIZE);                      
                            
                            
    d_C = (float *)(mxGPUGetData(C));
   // deviceC = (float *)(mxGPUGetData(C));
 //for (int i = 0; i < batch_count; i++) {
  //deviceA[i] = (float*) (d_A + i * numARows*numAColumns);
 // deviceB[i] = (float*) (d_B + i * numBRows*numBColumns);
 // deviceC[i] = (float*) (d_C + i * numCRows*numCColumns);// so here
//  deviceA[i] = (float*)(d_A);
//  deviceB[i] = (float*)(d_B);
 // deviceC[i] = (float*)((char*)d_C);
// }

	for (int i = 0; i<batch_count; i++) {

		deviceA[i] = (float*) ((char*) d_A + i * ((size_t) numARows*numAColumns) * sizeof(float));
		deviceB[i] = (float*) ((char*) d_B + i * ((size_t) numBRows*numBColumns) * sizeof(float));
		deviceC[i] = (float*) ((char*) d_C + i * ((size_t) numCRows*numCColumns) * sizeof(float));
        
   		//deviceA[i] = (float*) ( d_A + i *  numARows*numAColumns );
		//deviceB[i] = (float*) ( d_B + i *  numBRows*numBColumns );
		//deviceC[i] = (float*) ( d_C + i *  numCRows*numCColumns );
   
	}

   ThreeDMultiplicationCUBlas( numARows,
	 numAColumns,  numBRows,
	 numBColumns,  numCRows,  numCColumns,
	 batch_count,
	  deviceA,
	  deviceB,
	  deviceC,
	 alpha,
	 beta);

  //	cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

	//cudaDeviceSynchronize();//To synchronize the device
//Print_Mat(int Row, int Col, int BA,  float ** Mat)
      plhs[0] = mxGPUCreateMxArrayOnGPU(C);

      
      
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
   // mxFree(dimsA);
   // mxFree(dimsB);
   	//for (int i = 0; i<batch_count; i++) {
	//	free(deviceA[i]);
	//	free(deviceB[i]);
	//	free(deviceC[i]);
		//cudaFree(h_d_A[i]);
		//cudaFree(h_d_B[i]);
		//cudaFree(h_d_C[i]);
	//}

	free(deviceA);
	free(deviceB);
	free(deviceC);
     
	}

  else if ((!(mxIsGPUArray(prhs[0]))) && (!(mxIsGPUArray(prhs[1])))){
 
            
    int numARows, numAColumns, numBRows,  numBColumns, numCRows,  numCColumns;
    float alpha= mxGetScalar(prhs[2]);
    float beta = 0.0;
    
    const mwSize *dimsA;
    dimsA=mxGetDimensions(prhs[0]);
   numARows = (int)dimsA[0]; /* gets number of rows of A */
   numAColumns = (int)dimsA[1]; /* gets number of columns of A */
   int batch_count = (int)dimsA[2];
     const mwSize *dimsB;
    dimsB=mxGetDimensions(prhs[1]);
   numBRows = (int)dimsB[0]; /* gets number of rows of A */
   numBColumns = (int)dimsB[1]; /* gets number of columns of A */
     int batch_countB= (int)dimsB[2];
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
     numCRows = numARows;
	 numCColumns = numBColumns;
   if ((batch_count != batch_countB)||(numAColumns != numBRows)){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Array dimensions must match, the column number of first argument must be equal to the row number of second argument and the number of matrices must match.\n");
    }
     if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be float\n");
    }
    if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be float\n");
    }
    //hostB = (float *)mxGetData(prhs[1]);
    
    float **deviceA, **deviceB, **deviceC;
	deviceA = (float**)malloc(batch_count*sizeof(float*));
	deviceB = (float**)malloc(batch_count*sizeof(float*));
	deviceC = (float**)malloc(batch_count*sizeof(float*));
    //	for (int i = 0; i<batch_count; i++) {

	//	deviceA[i] = (float*)malloc(numARows*numAColumns*sizeof(float));
	//	deviceB[i] = (float*)malloc(numBRows*numBColumns*sizeof(float));
	//	deviceC[i] = (float*)malloc(numCRows*numCColumns*sizeof(float));
//	}



       
   // plhs[0] = mxCreateNumericMatrix(numCRows, numCColumns, mxSINGLE_CLASS, mxREAL);
    size_t pivot_dimensions[3] = {numCRows,numCColumns,batch_count};
     plhs[0] = mxCreateNumericArray(3, (mwSize*) pivot_dimensions,
       mxSINGLE_CLASS,  mxREAL);
    //float  *pointer = mxGetPr(plhs[0]);
    float  *d_A;
    float  *d_B;
    float *d_C;
    d_A = (float*) mxGetData(prhs[0]);
    d_B = (float*) mxGetData(prhs[1]);
    d_C = (float *)(mxGetData(plhs[0]));
    	for (int i = 0; i<batch_count; i++) {

		deviceA[i] = (float*) ((char*) d_A + i * ((size_t) numARows*numAColumns) * sizeof(float));
		deviceB[i] = (float*) ((char*) d_B + i * ((size_t) numBRows*numBColumns) * sizeof(float));
		deviceC[i] = (float*) ((char*) d_C + i * ((size_t) numCRows*numCColumns) * sizeof(float));
        
   		//deviceA[i] = (float*) ( d_A + i *  numARows*numAColumns );
		//deviceB[i] = (float*) ( d_B + i *  numBRows*numBColumns );
		//deviceC[i] = (float*) ( d_C + i *  numCRows*numCColumns );
   
	}
    
    //CalculateTransform(float * A, float * C, int numARows, int numAColumns, int numCRows, int numCColumns)  
    
     
     ThreeDMultiplicationCUBlasNGA( numARows,
	 numAColumns,  numBRows,
	 numBColumns,  numCRows,  numCColumns,
	 batch_count,
	  deviceA,
	  deviceB,
	  deviceC,
	 alpha,
	 beta);
    //mxDestroyArray(hostA);
    //mxDestroyArray(hostB);
    //mxFree(hostA);
    //mxFree(hostB);
    

        
        }//
     else if ((!mxIsNumeric(prhs[0]))||(!mxIsNumeric(prhs[1]))){
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments (input is not a numeric array)! \n");    
        }  


    
  
}
