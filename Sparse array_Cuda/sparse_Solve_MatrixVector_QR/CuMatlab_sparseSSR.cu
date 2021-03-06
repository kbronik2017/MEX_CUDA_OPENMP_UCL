
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
 * Sparse/Dense matrix-sparse/dense vector multiplication   Z=CuMatlab_solve(Sparse/Dense(A),Sparse/Dense(Y)).
 * AZ=Y -->Z=A\Y
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
#include <cusolverSp.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"
#include <omp.h>

// Input Arguments
#define	INPUTSPARSEA   prhs[0]
#define	INPUTSPARSEB   prhs[1]

// Output Arguments
#define	OUTPUTMATRIX  plhs[0]



  
    
extern "C" static void mexCuMatlab_sparseSSR(int nlhs, mxArray *plhs[],
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

    char const * const InputErrMsg = "Invalid input to MEX file, number of input arguments must be two.";
    char const * const OutputErrMsg = "Invalid output to MEX file, number of output arguments must be one.";
   if ((nrhs!=2)) {
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

	
	    if ( numARows < numAColumns ) {
       
              mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);  
       
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,first argument must be a sparse/dense tall (numARows > numAColumns) or square matrix.");
             
    } 
     if ( (numBColumns!= 1) ) {
         
              mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a dense/sparse column vector.");
             
    }
    if ( (numBRows!= numARows) ) {
              mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, array (matrix-vector) dimensions must agree.");
             
    }
	
	

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
      
 
    double  *VALSORTA = (double *)mxGetDoubles(VLSXY1);
    
    
    mwIndex nnz2;
    mxArray * VLSXY2 = mxGPUCreateMxArrayOnCPU(INPUTSPARSEGPUB);
    nnz2 = *(mxGetJc(VLSXY2) + numBColumns);

 
    int nnzB= static_cast<int> (nnz2);         
    
    mxArray *row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    int *pointerrowB = (int *)mxGetInt32s(row_sortB);
   
    Ir_DataGetSetIXY(VLSXY2 , pointerrowB, nnzB);
  
    
    double  *pointervalB = (double *)mxGetDoubles(VLSXY2);
            
   size_t pivot_dimensionsrowB[1] = {nnzB};
   
   size_t pivot_dimensionsvalueB[1] = {nnzB};    
      mxGPUArray *row_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrowB, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortB=(int *)mxGPUGetData(row_sortBB);
 gpuErrchk(cudaMemcpy(xrow_sortB, pointerrowB, nnzB * sizeof(*xrow_sortB), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueB, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        double *xval_sortB=(double*)mxGPUGetData(val_sortBB);
 gpuErrchk(cudaMemcpy(xval_sortB, pointervalB, nnzB * sizeof(*xval_sortB), cudaMemcpyHostToDevice)); 
   
 
         mxGPUDestroyGPUArray(INPUTSPARSEGPUB);
         mxDestroyArray(row_sortB);
         mxDestroyArray(VLSXY2);
		 
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
	
    size_t pivot_dimensionsvalueV[1] = {numBRows};

   mxGPUArray *DB_dense = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_B_dense = (double *)mxGPUGetData(DB_dense);
       cusparseSafeCall(cusparseDsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, d_B_dense, 
              CUSPARSE_INDEX_BASE_ONE));
			  
        mxGPUDestroyGPUArray(row_sortBB);
        mxGPUDestroyGPUArray(val_sortBB);			  
	
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   size_t pivot_dimensCOO_A[1] = {nnzA};
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(A);
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
         mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
         
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

	cusparseSafeCall(cusparseDgthr(handle, nnzA, d_A, d_A, Pa, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndA,
		nnzA,
		numARows,
		d_A_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));
    mxGPUDestroyGPUArray(COO_A);
    gpuErrchk(cudaFree(pBuffera));
	gpuErrchk(cudaFree(Pa));
    

const int batchSize = 1; 

cusolverSpHandle_t handle_cusolver; 
cusolverSpCreate(&handle_cusolver);
csrqrInfo_t info = NULL;

 size_t internalDataInBytes = 0;
 size_t workspaceInBytes = 0;

void *pBuffer = NULL; 

cusolverSafeCall(cusolverSpCreateCsrqrInfo(&info));


cusolverSafeCall(cusolverSpXcsrqrAnalysisBatched(
        handle_cusolver, numARows, numAColumns, nnzA,
        descrA, d_A_RowIndices, d_A_ColIndices,
        info));    
    
cusolverSafeCall(cusolverSpDcsrqrBufferInfoBatched(
    handle_cusolver, numARows, numAColumns, nnzA,
    descrA,
    d_A, d_A_RowIndices, d_A_ColIndices,
    batchSize, 
    info,
    &internalDataInBytes,
    &workspaceInBytes));

gpuErrchk(cudaMalloc(&pBuffer, workspaceInBytes));

    size_t pivot_dimensionsvalueVx[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueVx, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    double  *VALOUT = (double *)mxGPUGetData(VAL);

cusolverSafeCall(cusolverSpDcsrqrsvBatched(
    handle_cusolver, numARows, numAColumns, nnzA,
    descrA, d_A, d_A_RowIndices, d_A_ColIndices,
    d_B_dense,
    VALOUT,
    batchSize,
    info,
    pBuffer));
    
        mxGPUDestroyGPUArray(A);
        mxGPUDestroyGPUArray(ROW_A);
        mxGPUDestroyGPUArray(COL_A); 
        gpuErrchk(cudaFree(pBuffer));
        mxGPUDestroyGPUArray(DB_dense); 
OUTPUTMATRIX = mxGPUCreateMxArrayOnGPU(VAL);             

       
mxGPUDestroyGPUArray(VAL);
      
cusolverSpDestroyCsrqrInfo(info);
cusparseDestroyMatDescr(descrA);
cusolverSpDestroy(handle_cusolver);
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
           //     "Invalid input to MEX file, input(FIRST and SECOND  ARGUMENTS) must be  double precision.");
             
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
    if ( numARows < numAColumns ) {
      
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,first argument must be a sparse/dense tall (numARows > numAColumns) or square matrix.");
             
    } 
     if ( (numBColumns!= 1) ) {
  
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a dense/sparse column vector.");
             
    }
    if ( (numBRows!= numARows) ) {

         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, array (matrix-vector) dimensions must agree.");
             
    }

    
    mwIndex nnz1;
 
    nnz1 = *(mxGetJc(INPUTSPARSEA) + numAColumns);
    int nnzA = static_cast<int> (nnz1);
    
    mwIndex nnz2;
 
    nnz2 = *(mxGetJc(INPUTSPARSEB) + numBColumns);
    
 
   int nnzB= static_cast<int> (nnz2);
   
   
    mxArray *row_sortB =mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    int *pointerrowB = (int *)mxGetInt32s(row_sortB);
   
    Ir_DataGetSetIXY(INPUTSPARSEB , pointerrowB, nnzB);
    

    double  *pointervalB = (double *)mxGetDoubles(INPUTSPARSEB);        
   size_t pivot_dimensionsrowB[1] = {nnzB};
   
   size_t pivot_dimensionsvalueB[1] = {nnzB};    
      mxGPUArray *row_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrowB, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortB =(int *)mxGPUGetData(row_sortBB);
 gpuErrchk(cudaMemcpy(xrow_sortB, pointerrowB, nnzB * sizeof(*xrow_sortB), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortBB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueB, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        double *xval_sortB=(double*)mxGPUGetData(val_sortBB);
 gpuErrchk(cudaMemcpy(xval_sortB, pointervalB, nnzB * sizeof(*xval_sortB), cudaMemcpyHostToDevice));    
   
         	 
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
   
	         mxDestroyArray(row_sortB);	
  
   mxArray *  ROW_SORTA = mxCreateNumericMatrix(nnzA, 1,mxINT32_CLASS, mxREAL);
    int *ROWSORTA  = (int *)mxGetInt32s(ROW_SORTA);
       SetIr_Data(INPUTSPARSEA, ROWSORTA);

    
   mxArray *  COL_SORTA = mxCreateNumericMatrix(nnzA, 1, mxINT32_CLASS, mxREAL);
    int  *COLSORTA = (int *)mxGetInt32s(COL_SORTA);
          SetJc_Int(INPUTSPARSEA, COLSORTA);

      
    double  *VALSORTA = (double  *)mxGetDoubles(INPUTSPARSEA);
    
  
	
	size_t pivot_dimensionsvalueV[1] = {numBRows};

   mxGPUArray *DB_dense = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_B_dense = (double *)mxGPUGetData(DB_dense);
       cusparseSafeCall(cusparseDsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, d_B_dense, 
              CUSPARSE_INDEX_BASE_ONE));
			  
        mxGPUDestroyGPUArray(row_sortBB);
        mxGPUDestroyGPUArray(val_sortBB);
	
	
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   size_t pivot_dimensCOO_A[1] = {nnzA};
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(A);
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

	cusparseSafeCall(cusparseDgthr(handle, nnzA, d_A, d_A, Pa, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndA,
		nnzA,
		numARows,
		d_A_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));
    mxGPUDestroyGPUArray(COO_A);
    gpuErrchk(cudaFree(pBuffera));
	gpuErrchk(cudaFree(Pa));
    

const int batchSize = 1; 

cusolverSpHandle_t handle_cusolver; 
cusolverSpCreate(&handle_cusolver);
csrqrInfo_t info = NULL;

 size_t internalDataInBytes = 0;
 size_t workspaceInBytes = 0;

void *pBuffer = NULL; 

cusolverSafeCall(cusolverSpCreateCsrqrInfo(&info));


cusolverSafeCall(cusolverSpXcsrqrAnalysisBatched(
        handle_cusolver, numARows, numAColumns, nnzA,
        descrA, d_A_RowIndices, d_A_ColIndices,
        info));    
    
cusolverSafeCall(cusolverSpDcsrqrBufferInfoBatched(
    handle_cusolver, numARows, numAColumns, nnzA,
    descrA,
    d_A, d_A_RowIndices, d_A_ColIndices,
    batchSize, 
    info,
    &internalDataInBytes,
    &workspaceInBytes));

gpuErrchk(cudaMalloc(&pBuffer, workspaceInBytes));

    size_t pivot_dimensionsvalueVx[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueVx, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    double  *VALOUT = (double *)mxGPUGetData(VAL);
	

cusolverSafeCall(cusolverSpDcsrqrsvBatched(
    handle_cusolver, numARows, numAColumns, nnzA,
    descrA, d_A, d_A_RowIndices, d_A_ColIndices,
    d_B_dense,
    VALOUT,
    batchSize,
    info,
    pBuffer));
    
        mxGPUDestroyGPUArray(A);
        mxGPUDestroyGPUArray(ROW_A);
        mxGPUDestroyGPUArray(COL_A); 
        gpuErrchk(cudaFree(pBuffer));	
        mxGPUDestroyGPUArray(DB_dense);  
OUTPUTMATRIX = mxGPUCreateMxArrayOnGPU(VAL);             

       
mxGPUDestroyGPUArray(VAL);
      
cusolverSpDestroyCsrqrInfo(info);
cusparseDestroyMatDescr(descrA);
cusolverSpDestroy(handle_cusolver);
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
