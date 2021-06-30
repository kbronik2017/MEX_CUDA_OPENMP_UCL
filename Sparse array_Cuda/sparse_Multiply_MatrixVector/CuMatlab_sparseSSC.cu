
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
 * Sparse/Dense matrix-sparse/dense vector multiplication   Z=CuMatlab_multiplyV(Sparse/Dense(X),Sparse/Dense(Y), alpha).
 * Z= alpha*X*Y
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

#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"
#include <omp.h>

// Input Arguments
#define	INPUTSPARSEA   prhs[0]
#define	INPUTSPARSEB   prhs[1]
#define	ALPHA   prhs[2]
//#define	BETA    prhs[3]
// Output Arguments
#define	OUTPUTMATRIX  plhs[0]



  
    
extern "C" static void mexCuMatlab_sparseSSC(int nlhs, mxArray *plhs[],
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
    char const * const OutputErrMsg = "Invalid output to MEX file, number of output arguments must be one.";
   if ((nrhs!=3)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }
   if ((nlhs!=1)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", OutputErrMsg);
    }
 char *input_buf0;
 input_buf0 = mxArrayToString(INPUTSPARSEA);

      if ((mxIsChar(INPUTSPARSEA))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
    
 char *input_buf1;
 input_buf1 = mxArrayToString(INPUTSPARSEB);

      if ((mxIsChar(INPUTSPARSEB))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf1);
    } 

 char *input_buf2;
 input_buf2 = mxArrayToString(ALPHA);

      if ((mxIsChar(ALPHA))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(THIRD ARGUMENT) must be scalar not  %s\n",input_buf2);
    } 

if (mxIsGPUArray(INPUTSPARSEA) && mxIsGPUArray(INPUTSPARSEB)) {
    
    mxGPUArray const *INPUTSPARSEGPUA;
    mxGPUArray const *INPUTSPARSEGPUB;
    
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    INPUTSPARSEGPUA = mxGPUCreateFromMxArray(INPUTSPARSEA);
    INPUTSPARSEGPUB = mxGPUCreateFromMxArray(INPUTSPARSEB);
    
   
	
    if((mxGPUIsSparse(INPUTSPARSEGPUA))&& (mxGPUIsSparse(INPUTSPARSEGPUB)) ){
        
    const mwSize *dimsGPUSA;
    dimsGPUSA=mxGPUGetDimensions(INPUTSPARSEGPUA);
    int numARows, numAColumns;
    numARows = (int)dimsGPUSA[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPUSA[1]; /* gets number of columns of A */
    
    const mwSize *dimsGPUSB;
    dimsGPUSB=mxGPUGetDimensions(INPUTSPARSEGPUB);
    int numBRows, numBColumns;
    numBRows = (int)dimsGPUSB[0]; /* gets number of rows of B */
    numBColumns = (int)dimsGPUSB[1]; /* gets number of columns of B */
   if ( (numBColumns!= 1) ) {
              mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a sparse column vector.");
             
    }
    
    if ( (numAColumns!= numBRows) ) {
              mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);     
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, column number of sparse matrix(first argument) must be equal to row numbers of sparse vector(second argument).");
             
    }
        mxComplexDouble*  al= (mxComplexDouble *)mxGetComplexDoubles(ALPHA);
       const cuDoubleComplex alpha = make_cuDoubleComplex(al[0].real, al[0].imag);
       //mxComplexDouble*  bl= (mxComplexDouble *)mxGetComplexDoubles(BETA);
       const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
	   
	   
    mwIndex nnz1;
     mxArray * VLSXY1 = mxGPUCreateMxArrayOnCPU(INPUTSPARSEGPUA);
    nnz1 = *(mxGetJc(VLSXY1) + numAColumns);
    int nnzA = static_cast<int> (nnz1);
    
   
   mxArray *  ROW_SORTA = mxCreateNumericMatrix(nnzA, 1,mxINT32_CLASS, mxREAL);
    int *ROWSORTA  = (int *)mxGetInt32s(ROW_SORTA);
       SetIr_Data(VLSXY1, ROWSORTA);
    
   mxArray *  COL_SORTA = mxCreateNumericMatrix(nnzA, 1, mxINT32_CLASS, mxREAL);
    int  *COLSORTA = (int *)mxGetInt32s(COL_SORTA);
          SetJc_Int(VLSXY1, COLSORTA);
      
 
    cuDoubleComplex  *VALSORTA = (cuDoubleComplex *)mxGetComplexDoubles(VLSXY1);
    
    
    mwIndex nnz2;
    mxArray * VLSXY2 = mxGPUCreateMxArrayOnCPU(INPUTSPARSEGPUB);
    nnz2 = *(mxGetJc(VLSXY2) + numBColumns);
 
     
    int nnzB= static_cast<int> (nnz2);    
                 
  
    mxArray *row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    int *pointerrowB = (int *)mxGetInt32s(row_sortB);
   
    Ir_DataGetSetIXY(VLSXY2 , pointerrowB, nnzB);
  
 
   
    cuDoubleComplex  *pointervalB = (cuDoubleComplex *)mxGetComplexDoubles(VLSXY2);
            
   size_t pivot_dimensionsrowB[1] = {nnzB};
   
   size_t pivot_dimensionsvalueB[1] = {nnzB};    
      mxGPUArray *row_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrowB, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortB=(int *)mxGPUGetData(row_sortBB);
 gpuErrchk(cudaMemcpy(xrow_sortB, pointerrowB, nnzB * sizeof(*xrow_sortB), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueB, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortB=(cuDoubleComplex*)mxGPUGetData(val_sortBB);
 gpuErrchk(cudaMemcpy(xval_sortB, pointervalB, nnzB * sizeof(*xval_sortB), cudaMemcpyHostToDevice));    
    

         mxGPUDestroyGPUArray(INPUTSPARSEGPUB);
         mxDestroyArray(row_sortB);
         mxDestroyArray(VLSXY2); 
		 
		 
           
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
   
   
   
    size_t pivot_dimensionsvalueV[1] = {numBRows};

    mxGPUArray *dB_dense = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *d_B_dense = (cuDoubleComplex *)mxGPUGetData(dB_dense);
 
     cusparseSafeCall(cusparseZsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, d_B_dense, 
              CUSPARSE_INDEX_BASE_ONE));
 
          mxGPUDestroyGPUArray(row_sortBB);
          mxGPUDestroyGPUArray(val_sortBB);
 
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   size_t pivot_dimensCOO_A[1] = {nnzA};
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *d_A = (cuDoubleComplex *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);
    mxGPUArray * COO_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOO_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_cooRowIndA = (int *)mxGPUGetData(COO_A); 
	
	// --- Descriptor for sparse matrix B
	gpuErrchk(cudaMemcpy(d_A, VALSORTA, nnzA * sizeof(*d_A), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A_ColIndices, COLSORTA, nnzA * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cooRowIndA, ROWSORTA, nnzA * sizeof(*d_cooRowIndA), cudaMemcpyHostToDevice));
    
         mxDestroyArray(COL_SORTA);
         mxDestroyArray(ROW_SORTA);
         mxDestroyArray(VLSXY1);
         
         
	int *Pa = NULL;
	void *pBuffera = NULL;
	size_t pBufferSizeInBytesa = 0;
	cusparseXcoosort_bufferSizeExt(handle, numARows, numAColumns,
		nnzA,
		d_cooRowIndA,
		d_A_ColIndices, &pBufferSizeInBytesa);

	gpuErrchk(cudaMalloc(&pBuffera, sizeof(char)*pBufferSizeInBytesa));
	gpuErrchk(cudaMalloc(&Pa, sizeof(int)*nnzA));
	cusparseCreateIdentityPermutation(handle, nnzA, Pa);
	cusparseSafeCall(cusparseXcoosortByRow(handle, numARows, numAColumns,
		nnzA,
		d_cooRowIndA,
		d_A_ColIndices,
		Pa,
		pBuffera));

	cusparseSafeCall(cusparseZgthr(handle, nnzA, d_A, d_A, Pa, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndA,
		nnzA,
		numARows,
		d_A_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));
    mxGPUDestroyGPUArray(COO_A);
    gpuErrchk(cudaFree(pBuffera));
	gpuErrchk(cudaFree(Pa));
    
 
    size_t pivot_dimensionsvalue[1] = {numARows};
    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
    cuDoubleComplex  *VALOUT = (cuDoubleComplex *)mxGPUGetData(VAL);

 cusparseSafeCall(cusparseZcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
               numARows, numAColumns, nnzA, &alpha, 
               descrA, 
               d_A, 
               d_A_RowIndices, d_A_ColIndices,
               d_B_dense, &beta, 
               VALOUT));
          mxGPUDestroyGPUArray(A);  
          mxGPUDestroyGPUArray(ROW_A); 
          mxGPUDestroyGPUArray(COL_A);            
          mxGPUDestroyGPUArray(dB_dense);     
  OUTPUTMATRIX = mxGPUCreateMxArrayOnGPU(VAL);             

       
        mxGPUDestroyGPUArray(VAL);
  
        cusparseDestroyMatDescr(descrA);  
		cusparseDestroy(handle); 
  
        
        }
    
        else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }
    
   }
     
////////////////////////////////////////////////////////////////////////////////////  
    else if (!(mxIsGPUArray(INPUTSPARSEA)) && !(mxIsGPUArray(INPUTSPARSEB))){
   
     // if ((mxGetClassID(INPUTSPARSEA) != mxDOUBLE_CLASS) || (mxGetClassID(INPUTSPARSEB) != mxDOUBLE_CLASS)) {
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
           //     "Invalid input to MEX file, input(FIRST and SECOND  ARGUMENTS) must be  cuDoubleComplex precision.");
             
   // }
    if((mxIsSparse(INPUTSPARSEA))&& (mxIsSparse(INPUTSPARSEB)) ){
    
     mxInitGPU();
    const mwSize *dimsCPUA;
    dimsCPUA=mxGetDimensions(INPUTSPARSEA);
    
    int  numARows = (int)dimsCPUA[0]; /* gets number of rows of A */
    int  numAColumns = (int)dimsCPUA[1]; /* gets number of columns of A */
   
    const mwSize *dimsCPUB;
    dimsCPUB=mxGetDimensions(INPUTSPARSEB);
    
    int  numBRows = (int)dimsCPUB[0]; /* gets number of rows of B */
    int  numBColumns = (int)dimsCPUB[1]; /* gets number of columns of B */
   if ( (numBColumns!= 1) ) {
                
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a sparse column vector.");
             
    }
    
    if ( (numAColumns!= numBRows) ) {
        
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, column number of sparse matrix(first argument) must be equal to row numbers of sparse vector(second argument).");
             
    }
        mxComplexDouble*  al= (mxComplexDouble *)mxGetComplexDoubles(ALPHA);
       const cuDoubleComplex alpha = make_cuDoubleComplex(al[0].real, al[0].imag);
       //mxComplexDouble*  bl= (mxComplexDouble *)mxGetComplexDoubles(BETA);
       const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    mwIndex nnz1;
 
    nnz1 = *(mxGetJc(INPUTSPARSEA) + numAColumns);
    int nnzA = static_cast<int> (nnz1);
    
    mwIndex nnz2;
 
    nnz2 = *(mxGetJc(INPUTSPARSEB) + numBColumns);
    
 
      int nnzB= static_cast<int> (nnz2);
    

    mxArray *row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    int *pointerrowB = (int *)mxGetInt32s(row_sortB);
   
    Ir_DataGetSetIXY(INPUTSPARSEB , pointerrowB, nnzB);
    

   
    cuDoubleComplex  *pointervalB = (cuDoubleComplex *)mxGetComplexDoubles(INPUTSPARSEB);
            
   size_t pivot_dimensionsrowB[1] = {nnzB};
   
   size_t pivot_dimensionsvalueB[1] = {nnzB};    
      mxGPUArray *row_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrowB, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortB=(int *)mxGPUGetData(row_sortBB);
 gpuErrchk(cudaMemcpy(xrow_sortB, pointerrowB, nnzB * sizeof(*xrow_sortB), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueB, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortB=(cuDoubleComplex*)mxGPUGetData(val_sortBB);
 gpuErrchk(cudaMemcpy(xval_sortB, pointervalB, nnzB * sizeof(*xval_sortB), cudaMemcpyHostToDevice));    
   
         
         mxDestroyArray(row_sortB);    
	
	
  
   mxArray *  ROW_SORTA = mxCreateNumericMatrix(nnzA, 1,mxINT32_CLASS, mxREAL);
    int *ROWSORTA  = (int *)mxGetInt32s(ROW_SORTA);
       SetIr_Data(INPUTSPARSEA, ROWSORTA);

    
   mxArray *  COL_SORTA = mxCreateNumericMatrix(nnzA, 1, mxINT32_CLASS, mxREAL);
    int  *COLSORTA = (int *)mxGetInt32s(COL_SORTA);
          SetJc_Int(INPUTSPARSEA, COLSORTA);

      
    cuDoubleComplex  *VALSORTA = (cuDoubleComplex  *)mxGetComplexDoubles(INPUTSPARSEA);
    
  
 

    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
	
	
    size_t pivot_dimensionsvalueV[1] = {numBRows};

    mxGPUArray *DB_dense = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *d_B_dense = (cuDoubleComplex *)mxGPUGetData(DB_dense);
     cusparseSafeCall(cusparseZsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, d_B_dense, 
              CUSPARSE_INDEX_BASE_ONE));
			  
          mxGPUDestroyGPUArray(row_sortBB);
          mxGPUDestroyGPUArray(val_sortBB);
	
	
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   size_t pivot_dimensCOO_A[1] = {nnzA};
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *d_A = (cuDoubleComplex *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);
    mxGPUArray * COO_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOO_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_cooRowIndA = (int *)mxGPUGetData(COO_A); 
	
	// --- Descriptor for sparse matrix B
	gpuErrchk(cudaMemcpy(d_A, VALSORTA, nnzA * sizeof(*d_A), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A_ColIndices, COLSORTA, nnzA * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cooRowIndA, ROWSORTA, nnzA * sizeof(*d_cooRowIndA), cudaMemcpyHostToDevice));
    
         mxDestroyArray(COL_SORTA);
         mxDestroyArray(ROW_SORTA);
         
         
         
	int *Pa = NULL;
	void *pBuffera = NULL;
	size_t pBufferSizeInBytesa = 0;
	cusparseXcoosort_bufferSizeExt(handle, numARows, numAColumns,
		nnzA,
		d_cooRowIndA,
		d_A_ColIndices, &pBufferSizeInBytesa);

	gpuErrchk(cudaMalloc(&pBuffera, sizeof(char)*pBufferSizeInBytesa));
	gpuErrchk(cudaMalloc(&Pa, sizeof(int)*nnzA));
	cusparseCreateIdentityPermutation(handle, nnzA, Pa);
	cusparseSafeCall(cusparseXcoosortByRow(handle, numARows, numAColumns,
		nnzA,
		d_cooRowIndA,
		d_A_ColIndices,
		Pa,
		pBuffera));

	cusparseSafeCall(cusparseZgthr(handle, nnzA, d_A, d_A, Pa, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndA,
		nnzA,
		numARows,
		d_A_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));
    mxGPUDestroyGPUArray(COO_A);
    gpuErrchk(cudaFree(pBuffera));
	gpuErrchk(cudaFree(Pa));
 
    size_t pivot_dimensionsvalue[1] = {numARows};
    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
    cuDoubleComplex  *VALOUT = (cuDoubleComplex *)mxGPUGetData(VAL);

 cusparseSafeCall(cusparseZcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
               numARows, numAColumns, nnzA, &alpha, 
               descrA, 
               d_A, 
               d_A_RowIndices, d_A_ColIndices,
               d_B_dense, &beta, 
               VALOUT));
          mxGPUDestroyGPUArray(A);  
          mxGPUDestroyGPUArray(ROW_A); 
          mxGPUDestroyGPUArray(COL_A);            
          mxGPUDestroyGPUArray(DB_dense);    
  OUTPUTMATRIX = mxGPUCreateMxArrayOnGPU(VAL);             

       
        mxGPUDestroyGPUArray(VAL);
  
        cusparseDestroyMatDescr(descrA);  
		cusparseDestroy(handle);
    
    }
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }
    
 }
        //
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
