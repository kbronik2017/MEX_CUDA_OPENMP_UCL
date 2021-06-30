
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
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"
#include <omp.h>

// Input Arguments
#define	INPUTDENSEA   prhs[0]
#define	INPUTSPARSEB   prhs[1]


// Output Arguments
#define	OUTPUTMATRIX  plhs[0]



  
    
extern "C" static void mexCuMatlab_sparseDSC(int nlhs, mxArray *plhs[],
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
 input_buf0 = mxArrayToString(INPUTDENSEA);

      if ((mxIsChar(INPUTDENSEA))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
    
 char *input_buf1;
 input_buf1 = mxArrayToString(INPUTSPARSEB);

      if ((mxIsChar(INPUTSPARSEB))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf1);
    } 


if (mxIsGPUArray(INPUTDENSEA) && mxIsGPUArray(INPUTSPARSEB)) {
    
    mxGPUArray const *INPUTDENSEGPUA;
    mxGPUArray const *INPUTSPARSEGPUB;
    
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    INPUTDENSEGPUA = mxGPUCreateFromMxArray(INPUTDENSEA);
    INPUTSPARSEGPUB = mxGPUCreateFromMxArray(INPUTSPARSEB);
    
   
	
    if((!mxGPUIsSparse(INPUTDENSEGPUA))&& (mxGPUIsSparse(INPUTSPARSEGPUB)) ){
        
    const mwSize *dimsGPUSA;
    dimsGPUSA=mxGPUGetDimensions(INPUTDENSEGPUA);
    int numARows, numAColumns;
    numARows = (int)dimsGPUSA[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPUSA[1]; /* gets number of columns of A */
    
    const mwSize *dimsGPUSB;
    dimsGPUSB=mxGPUGetDimensions(INPUTSPARSEGPUB);
    int numBRows, numBColumns;
    numBRows = (int)dimsGPUSB[0]; /* gets number of rows of B */
    numBColumns = (int)dimsGPUSB[1]; /* gets number of columns of B */
	    if ( numARows != numAColumns ) {
       
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);    
       
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,first argument must be a sparse/dense square matrix.");
             
    } 
     if ( (numBColumns!= 1) ) {
         
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB); 
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a dense/sparse column vector.");
             
    }
    if ( (numBRows!= numARows) ) {
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB); 
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, array (matrix-vector) dimensions must agree.");
             
    }

   
    cuDoubleComplex const *d_A_dense;
   d_A_dense = (cuDoubleComplex const *)(mxGPUGetDataReadOnly(INPUTDENSEGPUA));
    
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
      
   
    size_t pivot_dimensionsvalueV[1] = {numBRows};

    mxGPUArray *dB_dense = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *d_B_dense = (cuDoubleComplex *)mxGPUGetData(dB_dense);
 
     cusparseSafeCall(cusparseZsctr(handle, nnzB, 
              xval_sortB, 
              xrow_sortB, d_B_dense, 
              CUSPARSE_INDEX_BASE_ONE));
			  
          mxGPUDestroyGPUArray(row_sortBB);
          mxGPUDestroyGPUArray(val_sortBB);			  
    
    int nnzA = 0;                            // --- Number of nonzero elements in dense matrix A
	const int lda = numARows;
	//int *d_nnzPerVectorA;   // gpuErrchk(cudaMalloc(&d_nnzPerVectorA, numARows * sizeof(*d_nnzPerVectorA)));
	
	size_t pivot_pervect[1] = {numARows};
    mxGPUArray *PerVect = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect);
	//cuDoubleComplex *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
	//gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	cusparseSafeCall(cusparseZnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));      
       // cuDoubleComplex *d_A;           // gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
		//int *d_A_RowIndices;    //gpuErrchk(cudaMalloc(&d_A_RowIndices, (numARows + 1) * sizeof(*d_A_RowIndices)));
		//int *d_A_ColIndices;    //gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
		
		
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *d_A = (cuDoubleComplex *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);

		cusparseSafeCall(cusparseZdense2csr(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));       
        //gpuErrchk(cudaFree(d_A_dense));
         mxGPUDestroyGPUArray(PerVect);
         mxGPUDestroyGPUArray(INPUTDENSEGPUA);

 cusolverSpHandle_t handle_cusolver; 
cusolverSpCreate(&handle_cusolver);
csrcholInfo_t chl_info = NULL;
    const double tol = 1.e-14; 
    int singularity = 0;
    size_t size_internal = 0; 
    size_t size_chol = 0;
  cusolverSafeCall(cusolverSpCreateCsrcholInfo(&chl_info));           

  cusolverSafeCall(cusolverSpXcsrcholAnalysis(
        handle_cusolver, numARows, nnzA,
        descrA, d_A_RowIndices, d_A_ColIndices,
        chl_info));  
    
  cusolverSafeCall(cusolverSpZcsrcholBufferInfo(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        &size_internal,
        &size_chol));   
     
    void *buffer_gpu = NULL; 

    gpuErrchk(cudaMalloc(&buffer_gpu, sizeof(char)*size_chol)); 
   
     cusolverSafeCall(cusolverSpZcsrcholFactor(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        buffer_gpu));
    
    cusolverSafeCall(cusolverSpZcsrcholZeroPivot(
        handle_cusolver, chl_info, tol, &singularity));
    
    if ( 0 <= singularity){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                       "Invalid input to MEX file, (fatal error:) A is not invertible, singularity=%d\n", singularity);
       
    }
    
    
    size_t pivot_dimensionsvalueVa[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueVa, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
    cuDoubleComplex  *VALOUT = (cuDoubleComplex *)mxGPUGetData(VAL);

    cusolverSafeCall(cusolverSpZcsrcholSolve(
        handle_cusolver, numARows, d_B_dense, VALOUT, chl_info, buffer_gpu));
    

        mxGPUDestroyGPUArray(A);
        mxGPUDestroyGPUArray(ROW_A);
        mxGPUDestroyGPUArray(COL_A);   
        mxGPUDestroyGPUArray(dB_dense);
		
OUTPUTMATRIX = mxGPUCreateMxArrayOnGPU(VAL);             
gpuErrchk(cudaFree(buffer_gpu));       
mxGPUDestroyGPUArray(VAL);
cusolverSpDestroyCsrcholInfo(chl_info);      
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
    else if (!(mxIsGPUArray(INPUTDENSEA)) && !(mxIsGPUArray(INPUTSPARSEB))){
   
     // if ((mxGetClassID(INPUTSPARSEA) != mxDOUBLE_CLASS) || (mxGetClassID(INPUTSPARSEB) != mxDOUBLE_CLASS)) {
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
           //     "Invalid input to MEX file, input(FIRST and SECOND  ARGUMENTS) must be  cuDoubleComplex precision.");
             
   // }
    if((!mxIsSparse(INPUTDENSEA))&& (mxIsSparse(INPUTSPARSEB)) ){
    
     mxInitGPU();
    const mwSize *dimsCPUA;
    dimsCPUA=mxGetDimensions(INPUTDENSEA);
    
    int  numARows = (int)dimsCPUA[0]; /* gets number of rows of A */
    int  numAColumns = (int)dimsCPUA[1]; /* gets number of columns of A */
   
    const mwSize *dimsCPUB;
    dimsCPUB=mxGetDimensions(INPUTSPARSEB);
    
    int  numBRows = (int)dimsCPUB[0]; /* gets number of rows of B */
    int  numBColumns = (int)dimsCPUB[1]; /* gets number of columns of B */
    if ( numARows != numAColumns  ) {
      
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,first argument must be a sparse/dense square matrix.");
             
    } 
     if ( (numBColumns!= 1) ) {
  
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a dense/sparse column vector.");
             
    }
    if ( (numBRows!= numARows) ) {

         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, array (matrix-vector) dimensions must agree.");
             
    }
    
 
    
   cuDoubleComplex *h_A_dense1;
   h_A_dense1 = (cuDoubleComplex *)mxGetComplexDoubles(INPUTDENSEA);
    
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
		  
		  
    int nnzA = 0;                            // --- Number of nonzero elements in dense matrix A
	const int lda = numARows;
	//int *d_nnzPerVectorA;    gpuErrchk(cudaMalloc(&d_nnzPerVectorA, numARows * sizeof(*d_nnzPerVectorA)));
	size_t pivot_pervect[1] = {numARows};
    mxGPUArray *PerVect = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect);
	
	//cuDoubleComplex *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
	
	  size_t pivot_dimensionsvalueDA[2] = {numARows, numAColumns};
      mxGPUArray *OUTMA = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsvalueDA, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
      cuDoubleComplex *d_A_dense = (cuDoubleComplex *)mxGPUGetData(OUTMA);
	
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	cusparseSafeCall(cusparseZnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));      
       // cuDoubleComplex *d_A;           // gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
		//int *d_A_RowIndices;   // gpuErrchk(cudaMalloc(&d_A_RowIndices, (numARows + 1) * sizeof(*d_A_RowIndices)));
		//int *d_A_ColIndices;   // gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *d_A = (cuDoubleComplex *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);
	

		cusparseSafeCall(cusparseZdense2csr(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));       
		  mxGPUDestroyGPUArray(OUTMA);
        //gpuErrchk(cudaFree(d_nnzPerVectorA));
          mxGPUDestroyGPUArray(PerVect);
    
cusolverSpHandle_t handle_cusolver; 
cusolverSpCreate(&handle_cusolver);
csrcholInfo_t chl_info = NULL;
    const double tol = 1.e-14; 
    int singularity = 0;
    size_t size_internal = 0; 
    size_t size_chol = 0;
  cusolverSafeCall(cusolverSpCreateCsrcholInfo(&chl_info));           

  cusolverSafeCall(cusolverSpXcsrcholAnalysis(
        handle_cusolver, numARows, nnzA,
        descrA, d_A_RowIndices, d_A_ColIndices,
        chl_info));  
    
  cusolverSafeCall(cusolverSpZcsrcholBufferInfo(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        &size_internal,
        &size_chol));   
     
    void *buffer_gpu = NULL; 

    gpuErrchk(cudaMalloc(&buffer_gpu, sizeof(char)*size_chol)); 
   
     cusolverSafeCall(cusolverSpZcsrcholFactor(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        buffer_gpu));
    
    cusolverSafeCall(cusolverSpZcsrcholZeroPivot(
        handle_cusolver, chl_info, tol, &singularity));
    
    if ( 0 <= singularity){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                       "Invalid input to MEX file, (fatal error:) A is not invertible, singularity=%d\n", singularity);
       
    }
    
    
    size_t pivot_dimensionsvalueVa[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueVa, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
    cuDoubleComplex  *VALOUT = (cuDoubleComplex *)mxGPUGetData(VAL);

    cusolverSafeCall(cusolverSpZcsrcholSolve(
        handle_cusolver, numARows, d_B_dense, VALOUT, chl_info, buffer_gpu));
    

        mxGPUDestroyGPUArray(A);
        mxGPUDestroyGPUArray(ROW_A);
        mxGPUDestroyGPUArray(COL_A);   
        mxGPUDestroyGPUArray(DB_dense);
		
OUTPUTMATRIX = mxGPUCreateMxArrayOnGPU(VAL);             
gpuErrchk(cudaFree(buffer_gpu));       
mxGPUDestroyGPUArray(VAL);
cusolverSpDestroyCsrcholInfo(chl_info);      
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
