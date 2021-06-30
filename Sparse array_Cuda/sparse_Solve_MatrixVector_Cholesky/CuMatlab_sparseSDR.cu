
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
#define	INPUTSPARSEA   prhs[0]
#define	INPUTDENSEB   prhs[1]

// Output Arguments
#define	OUTPUTMATRIX  plhs[0]



  
    
extern "C" static void mexCuMatlab_sparseSDR(int nlhs, mxArray *plhs[],
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
 input_buf1 = mxArrayToString(INPUTDENSEB);

      if ((mxIsChar(INPUTDENSEB))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf1);
    } 


if (mxIsGPUArray(INPUTSPARSEA) && mxIsGPUArray(INPUTDENSEB)) {
    
    mxGPUArray const *INPUTSPARSEGPUA;
    mxGPUArray const *INPUTDENSEGPUB;
    
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    INPUTSPARSEGPUA = mxGPUCreateFromMxArray(INPUTSPARSEA);
    INPUTDENSEGPUB = mxGPUCreateFromMxArray(INPUTDENSEB);
    
   
	
    if((mxGPUIsSparse(INPUTSPARSEGPUA))&& (!mxGPUIsSparse(INPUTDENSEGPUB)) ){
        
    const mwSize *dimsGPUSA;
    dimsGPUSA=mxGPUGetDimensions(INPUTSPARSEGPUA);
    int numARows, numAColumns;
    numARows = (int)dimsGPUSA[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPUSA[1]; /* gets number of columns of A */
    
    const mwSize *dimsGPUSB;
    dimsGPUSB=mxGPUGetDimensions(INPUTDENSEGPUB);
    int numBRows, numBColumns;
    numBRows = (int)dimsGPUSB[0]; /* gets number of rows of B */
    numBColumns = (int)dimsGPUSB[1]; /* gets number of columns of B */
    if ( numARows != numAColumns) {
       
        mxGPUDestroyGPUArray(INPUTDENSEGPUB);
        mxGPUDestroyGPUArray(INPUTSPARSEGPUA);   
       
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,first argument must be a sparse/dense square matrix.");
             
    } 
     if ( (numBColumns!= 1) ) {
         
        mxGPUDestroyGPUArray(INPUTDENSEGPUB);
        mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a dense/sparse column vector.");
             
    }
    if ( (numBRows!= numARows) ) {
        mxGPUDestroyGPUArray(INPUTDENSEGPUB);
        mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, array (matrix-vector) dimensions must agree.");
             
    }
    


    
    double const *d_B_dense;
   d_B_dense = (double const *)(mxGPUGetDataReadOnly(INPUTDENSEGPUB));
    

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
           
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);


		mxGPUDestroyGPUArray(INPUTSPARSEGPUA);
        mxGPUDestroyGPUArray(INPUTDENSEGPUB);
		
		
	//double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
	//int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (numARows + 1) * sizeof(*d_A_RowIndices)));
	//int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
	//int *d_cooRowIndA;       gpuErrchk(cudaMalloc(&d_cooRowIndA, nnzA * sizeof(*d_cooRowIndA)));
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
    
  cusolverSafeCall(cusolverSpDcsrcholBufferInfo(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        &size_internal,
        &size_chol));   
     
    void *buffer_gpu = NULL; 

    gpuErrchk(cudaMalloc(&buffer_gpu, sizeof(char)*size_chol)); 
   
     cusolverSafeCall(cusolverSpDcsrcholFactor(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        buffer_gpu));
    
    cusolverSafeCall(cusolverSpDcsrcholZeroPivot(
        handle_cusolver, chl_info, tol, &singularity));
    
    if ( 0 <= singularity){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                       "Invalid input to MEX file, (fatal error:) A is not invertible, singularity=%d\n", singularity);
       
    }
    
    
    size_t pivot_dimensionsvalueVa[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueVa, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    double  *VALOUT = (double *)mxGPUGetData(VAL);

    cusolverSafeCall(cusolverSpDcsrcholSolve(
        handle_cusolver, numARows, d_B_dense, VALOUT, chl_info, buffer_gpu));
    

        mxGPUDestroyGPUArray(A);
        mxGPUDestroyGPUArray(ROW_A);
        mxGPUDestroyGPUArray(COL_A);   
   
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
    else if (!(mxIsGPUArray(INPUTSPARSEA)) && !(mxIsGPUArray(INPUTDENSEB))){
   
     // if ((mxGetClassID(INPUTSPARSEA) != mxDOUBLE_CLASS) || (mxGetClassID(INPUTSPARSEB) != mxDOUBLE_CLASS)) {
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
           //     "Invalid input to MEX file, input(FIRST and SECOND  ARGUMENTS) must be  double precision.");
             
   // }
    if((mxIsSparse(INPUTSPARSEA))&& (!mxIsSparse(INPUTDENSEB)) ){
    
     mxInitGPU();
    const mwSize *dimsCPUA;
    dimsCPUA=mxGetDimensions(INPUTSPARSEA);
    
    int  numARows = (int)dimsCPUA[0]; /* gets number of rows of A */
    int  numAColumns = (int)dimsCPUA[1]; /* gets number of columns of A */
   
    const mwSize *dimsCPUB;
    dimsCPUB=mxGetDimensions(INPUTDENSEB);
    
    int  numBRows = (int)dimsCPUB[0]; /* gets number of rows of B */
    int  numBColumns = (int)dimsCPUB[1]; /* gets number of columns of B */
    if ( numARows != numAColumns ) {
      
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

      
    
    mwIndex nnz1;
 
    nnz1 = *(mxGetJc(INPUTSPARSEA) + numAColumns);
    int nnzA = static_cast<int> (nnz1);
    

   
   mxArray *  ROW_SORTA = mxCreateNumericMatrix(nnzA, 1,mxINT32_CLASS, mxREAL);
    int *ROWSORTA  = (int *)mxGetInt32s(ROW_SORTA);
       SetIr_Data(INPUTSPARSEA, ROWSORTA);

    
   mxArray *  COL_SORTA = mxCreateNumericMatrix(nnzA, 1, mxINT32_CLASS, mxREAL);
    int  *COLSORTA = (int *)mxGetInt32s(COL_SORTA);
          SetJc_Int(INPUTSPARSEA, COLSORTA);

      
    double  *VALSORTA = (double  *)mxGetDoubles(INPUTSPARSEA);
           
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);

	
	  size_t pivot_dimensionsvalueDB[1] = {numBRows};
      mxGPUArray *OUTMB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueDB, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_B_dense = (double *)mxGPUGetData(OUTMB);
	
   double *h_B_dense1;
   h_B_dense1 = (double *)mxGetDoubles(INPUTDENSEB);
	gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense1, numBRows * sizeof(*d_B_dense), cudaMemcpyHostToDevice));
    
    
    	//double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
	//int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (numARows + 1) * sizeof(*d_A_RowIndices)));
	//int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
	//int *d_cooRowIndA;       gpuErrchk(cudaMalloc(&d_cooRowIndA, nnzA * sizeof(*d_cooRowIndA)));
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
    
  cusolverSafeCall(cusolverSpDcsrcholBufferInfo(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        &size_internal,
        &size_chol));   
     
    void *buffer_gpu = NULL; 

    gpuErrchk(cudaMalloc(&buffer_gpu, sizeof(char)*size_chol)); 
   
     cusolverSafeCall(cusolverSpDcsrcholFactor(
        handle_cusolver, numARows, nnzA,
        descrA, d_A, d_A_RowIndices, d_A_ColIndices,
        chl_info,
        buffer_gpu));
    
    cusolverSafeCall(cusolverSpDcsrcholZeroPivot(
        handle_cusolver, chl_info, tol, &singularity));
    
    if ( 0 <= singularity){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                       "Invalid input to MEX file, (fatal error:) A is not invertible, singularity=%d\n", singularity);
       
    }
    
    size_t pivot_dimensionsvalueVa[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueVa, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    double  *VALOUT = (double *)mxGPUGetData(VAL);

    cusolverSafeCall(cusolverSpDcsrcholSolve(
        handle_cusolver, numARows, d_B_dense, VALOUT, chl_info, buffer_gpu));
    

        mxGPUDestroyGPUArray(A);
        mxGPUDestroyGPUArray(ROW_A);
        mxGPUDestroyGPUArray(COL_A);   
        mxGPUDestroyGPUArray(OUTMB);
		
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
