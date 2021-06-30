
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}[double/complex double]
 * Sparse/Dense --> Dense,   Z=CuMatlab_full(Sparse/Dense(X)).
 * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 * Wellcome Trust Centre for Neuroimaging
 * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 * Copyright 2018
 * Kevin Bronik
 */

#include "matrix.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cusparse_v2.h>


#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"

// Input Arguments
#define	INPUTMATRIX prhs[0]

#define	OUPUTMATRIX plhs[0]
// Output Arguments


   
extern "C" static void mexCuMatlab_fullD(int nlhs, mxArray *plhs[],
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

    char const * const InputErrMsg = "Invalid input to MEX file, number of input arguments must be one.";
    char const * const OutputErrMsg = "Invalid output to MEX file, number of output arguments must be one.";
   if ((nrhs!=1)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }
   if ((nlhs!=1)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", OutputErrMsg);
    }

 char *input_buf0;
 input_buf0 = mxArrayToString(INPUTMATRIX);

      if ((mxIsChar(INPUTMATRIX))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }



if ((mxIsGPUArray(INPUTMATRIX))) {
    mxInitGPU();
	mxGPUArray const *GPUINPUTMATRIX = mxGPUCreateFromMxArray(INPUTMATRIX);	
    
     if(mxGPUIsSparse(GPUINPUTMATRIX)==0) {  
        OUPUTMATRIX = mxDuplicateArray((mxGPUCreateMxArrayOnCPU(GPUINPUTMATRIX)));
       
          printf("Warning! Input(FIRST ARGUMENT) must be sparse!, continuing execution... \n");   
                return;
    }  
  //  if ((mxGetClassID(mxGPUCreateMxArrayOnCPU(GPUINPUTMATRIX))) != mxDOUBLE_CLASS) {
     //    mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
        //        "Invalid input to MEX file, input(FIRST ARGUMENT) must be double precision.");
             
   // }
    const mwSize *dimsGPU;
    dimsGPU=mxGPUGetDimensions(GPUINPUTMATRIX);
       
     int numARows, numAColumns;
     numARows = (int)dimsGPU[0]; /* gets number of rows of A */
     numAColumns = (int)dimsGPU[1]; /* gets number of columns of A */

     mwIndex nnz1;
    
    mxArray * tempx = mxGPUCreateMxArrayOnCPU(GPUINPUTMATRIX);
    nnz1 = *(mxGetJc(tempx) + numAColumns);
     //nnz1=(mwSize)ceil(numARows*numAColumns);
    int nnz= static_cast<int> (nnz1);

    mxArray *row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    int *pointerrow = (int *)mxGetInt32s(row_sort);
   
    Ir_DataGetSet(tempx , pointerrow, nnz);

    mxArray *col_sort =mxCreateNumericMatrix(numAColumns+1, 1, mxINT32_CLASS, mxREAL);
    int *pointercol = (int *)mxGetInt32s(col_sort);
    
    Jc_DataGetSet(tempx , pointercol, numAColumns);
     mxGPUDestroyGPUArray(GPUINPUTMATRIX);
    double  *pointerval = (double *)mxGetDoubles(tempx);
   size_t pivot_dimensionsrow[1] = {nnz};
   size_t pivot_dimensionscolumn[1] = {numAColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnz};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(VAL_SORT1);
    
    
    
    
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));


	//double *d_A_denseReconstructed; gpuErrchk(cudaMalloc(&d_A_denseReconstructed, numARows * numAColumns * sizeof(double)));
    mxGPUArray *denseReconstructed = mxGPUCreateGPUArray(2, dimsGPU, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double *d_A_denseReconstructed = (double*)mxGPUGetData(denseReconstructed);
		//double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
		//int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, nnz * sizeof(*d_A_RowIndices)));
		//int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, (numAColumns + 1) *sizeof(*d_A_ColIndices)));

		// --- Descriptor for sparse matrix A
		gpuErrchk(cudaMemcpy(d_A, pointerval, nnz * sizeof(*d_A), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_ColIndices, pointercol, (numAColumns + 1) * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_RowIndices, pointerrow, nnz *sizeof(*d_A_RowIndices), cudaMemcpyHostToDevice));

		cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSafeCall(cusparseDcsc2dense(handle, numARows, numAColumns, descrA, d_A, d_A_RowIndices, d_A_ColIndices,
			d_A_denseReconstructed, numARows));
	//gpuErrchk(cudaMemcpy(h_A_denseReconstructed, d_A_denseReconstructed, numARows * numAColumns * sizeof(double), cudaMemcpyDeviceToHost));    
        
    OUPUTMATRIX = mxGPUCreateMxArrayOnGPU(denseReconstructed);
   
    
    mxGPUDestroyGPUArray(denseReconstructed);
    mxDestroyArray(row_sort);
    mxDestroyArray(col_sort);
    mxDestroyArray(tempx); 
      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
    //gpuErrchk(cudaFree(d_A_denseReconstructed));

	cusparseDestroyMatDescr(descrA);  
	cusparseDestroy(handle);
    
  
    }
 
    else if (!(mxIsGPUArray(INPUTMATRIX))){
   
    if(!mxIsSparse(INPUTMATRIX)) {
        OUPUTMATRIX = mxDuplicateArray(INPUTMATRIX);
      
          printf("Warning! Input(FIRST ARGUMENT) must be sparse!, continuing execution... \n");   
                return;
    } 
            
     // if (mxGetClassID(INPUTMATRIX) != mxDOUBLE_CLASS) {
      //   mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
           //     "Invalid input to MEX file, input(FIRST ARGUMENT) must be double precision.");
             
  //  }

        
     int numARows, numAColumns;
     numARows = (int)mxGetM(INPUTMATRIX); 
     numAColumns = (int)mxGetN(INPUTMATRIX);
    
    mwIndex nnz1;
    
    nnz1 = *(mxGetJc(INPUTMATRIX) + numAColumns);
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    const mwSize *dimsGPU;
    dimsGPU=mxGetDimensions(INPUTMATRIX);
    
    int nnz= static_cast<int> (nnz1);
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
    double * h_A=(double *)mxGetDoubles(INPUTMATRIX);
 	int *h_A_ColIndices = Jc_Data(INPUTMATRIX);
	int *h_A_RowIndices = Ir_Data(INPUTMATRIX);
    
   // OUPUTMATRIX = mxCreateNumericMatrix(numARows, numAColumns, mxDOUBLE_CLASS, mxREAL);
   // double  *h_A_denseReconstructed = (double *)mxGetDoubles(OUPUTMATRIX);
    
   size_t pivot_dimensionsrow[1] = {nnz};
   size_t pivot_dimensionscolumn[1] = {numAColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnz};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
   int  *d_A_ColIndices = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(VAL_SORT1);
    mxGPUArray *denseReconstructed = mxGPUCreateGPUArray(2, dimsGPU, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double *d_A_denseReconstructed = (double*)mxGPUGetData(denseReconstructed);
    	gpuErrchk(cudaMemcpy(d_A, h_A, nnz * sizeof(*d_A), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_ColIndices, h_A_ColIndices, (numAColumns + 1) * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_RowIndices, h_A_RowIndices, nnz *sizeof(*d_A_RowIndices), cudaMemcpyHostToDevice));

		cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSafeCall(cusparseDcsc2dense(handle, numARows, numAColumns, descrA, d_A, d_A_RowIndices, d_A_ColIndices,
			d_A_denseReconstructed, numARows));
    
        
      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
      
    //gpuErrchk(cudaMemcpy(h_A_denseReconstructed, d_A_denseReconstructed, numARows * numAColumns * sizeof(double), cudaMemcpyDeviceToHost));
    OUPUTMATRIX = mxGPUCreateMxArrayOnGPU(denseReconstructed);
     mxGPUDestroyGPUArray(denseReconstructed);
        
		cusparseDestroyMatDescr(descrA);  
		cusparseDestroy(handle);
    
     }
        
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
