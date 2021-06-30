
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}[double or complex double]
 * Create row, column, value vectors from sparse/dense matrix  [row, column, value]=CuMatlab_find(Sparse/Dense(X))
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
#include <cusolverDn.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"

// Input Arguments
#define	INPUTMATRIX   prhs[0]


// Output Arguments
#define	LeftUnitaryMatrix plhs[0]
#define	SingularValueVector plhs[1]
#define	RightUnitaryMatrix plhs[2]

#define min(a,b) ((a) < (b) ? (a) : (b))
#define	mediumsizematrix   150000000                         
    
extern "C" static void mexCuMatlab_svdD(int nlhs, mxArray *plhs[],
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
    char const * const OutputErrMsg = "Invalid output to MEX file, number of output arguments must be three.";
   if ((nrhs!=1)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", InputErrMsg);
    }
   if ((nlhs!=3)) {
        mexErrMsgIdAndTxt("MATLAB:mexatexit:invalidInput", OutputErrMsg);
    }
 char *input_buf0;
 input_buf0 = mxArrayToString(INPUTMATRIX);

      if ((mxIsChar(INPUTMATRIX))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
    

if (mxIsGPUArray(INPUTMATRIX)) {
		
    
    mxGPUArray const *INPUTMATRIXGPU;
    
    int numARows, numAColumns;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    INPUTMATRIXGPU = mxGPUCreateFromMxArray(INPUTMATRIX);
   if(mxGPUIsSparse(INPUTMATRIXGPU)==1) {
      const mwSize *dimsGPU;
    dimsGPU=mxGPUGetDimensions(INPUTMATRIXGPU);
       
     int numARows, numAColumns;
     numARows = (int)dimsGPU[0]; /* gets number of rows of A */
     numAColumns = (int)dimsGPU[1]; /* gets number of columns of A */
    if ( numARows < numAColumns ) {
       
         mxGPUDestroyGPUArray(INPUTMATRIXGPU);   
       
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,(first) argument must be a sparse/dense tall (numARows > numAColumns) or square matrix.");
             
    } 
     mwIndex nnz1;
    
    mxArray * tempx = mxGPUCreateMxArrayOnCPU(INPUTMATRIXGPU);
    nnz1 = *(mxGetJc(tempx) + numAColumns);
     //nnz1=(mwSize)ceil(numARows*numAColumns);
    int nnz= static_cast<int> (nnz1);

    mxArray *row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    int *pointerrow = (int *)mxGetInt32s(row_sort);
   
    Ir_DataGetSet(tempx , pointerrow, nnz);

    mxArray *col_sort =mxCreateNumericMatrix(numAColumns+1, 1, mxINT32_CLASS, mxREAL);
    int *pointercol = (int *)mxGetInt32s(col_sort);
    
    Jc_DataGetSet(tempx , pointercol, numAColumns);
     mxGPUDestroyGPUArray(INPUTMATRIXGPU);
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
		cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

	//double *d_A_denseReconstructed; gpuErrchk(cudaMalloc(&d_A_denseReconstructed, numARows * numAColumns * sizeof(double)));
    mxGPUArray *denseReconstructed = mxGPUCreateGPUArray(2, dimsGPU, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double *d_A_dense = (double*)mxGPUGetData(denseReconstructed);
		//double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
		//int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, nnz * sizeof(*d_A_RowIndices)));
		//int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, (numAColumns + 1) *sizeof(*d_A_ColIndices)));

		// --- Descriptor for sparse matrix A
		gpuErrchk(cudaMemcpy(d_A, pointerval, nnz * sizeof(*d_A), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_ColIndices, pointercol, (numAColumns + 1) * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_RowIndices, pointerrow, nnz *sizeof(*d_A_RowIndices), cudaMemcpyHostToDevice));


		cusparseSafeCall(cusparseDcsc2dense(handle, numARows, numAColumns, descrA, d_A, d_A_RowIndices, d_A_ColIndices,
			d_A_dense, numARows));

      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);  
      // this uses QR algorithm (for large matrices)       
      if ( numARows*numAColumns >= mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	//cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
   
      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      
      cusolverSafeCall(cusolverDnDgesvd_bufferSize(cusolver_handle, numARows, numAColumns, &lwork));
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
         
      size_t pivot_dimensionsrwork[1] = {min(numARows, numAColumns)-1};
      mxGPUArray *RW = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrwork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *rwork = (double *)mxGPUGetData(RW); 
      
    cusolverSafeCall(cusolverDnDgesvd(cusolver_handle, 'A', 'A', numARows, numAColumns, d_A_dense, numARows,
    d_S, d_U, numARows, d_V, numAColumns, work, lwork, rwork, devInfo));
    
    mxGPUDestroyGPUArray(denseReconstructed);
    
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>0){
        printf("The unconverged superdiagonal elements of an upper bidiagonal matrix:");
        
       mxArray *rt =mxCreateNumericMatrix((min(numARows, numAColumns)-1), 1, mxDOUBLE_CLASS, mxREAL);
       double *rwork_host = (double *)mxGetDoubles(rt);
       
        gpuErrchk(cudaMemcpy(rwork_host, rwork, (min(numARows, numAColumns)-1)*sizeof(double), cudaMemcpyDeviceToHost));
     for (int i = 0; i < (min(numARows, numAColumns)-1); i++){
        printf("rwork_host[%i] = %i \n", i, rwork_host[i]);    
         }
         printf("\n");
         mxDestroyArray(rt);
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 

	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
    }
   
   // this uses the parallelism of Jacobi method (for small and medium size matrices) 
   if ( numARows*numAColumns < mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	//cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    
    const double tol = 1.e-7;
    const int max_sweeps = 150;
    const int econ = 0 ; 

      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      gesvdjInfo_t params= NULL;
      
      
    cusolverSafeCall(cusolverDnCreateGesvdjInfo(&params));


    cusolverSafeCall(cusolverDnXgesvdjSetTolerance(
        params,
        tol));
    

    cusolverSafeCall(cusolverDnXgesvdjSetMaxSweeps(
        params,
        max_sweeps));
    
      
 
    cusolverSafeCall(cusolverDnDgesvdj_bufferSize(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,      
    numARows,             
    d_S, 
    d_U,      
    numARows,              
    d_V,      
    numAColumns,              
    &lwork,
    params));
      
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
      
    cusolverSafeCall(cusolverDnDgesvdj(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,            
    numARows,              
    d_S,       
    d_U,            
    numARows,              
    d_V,            
    numAColumns,              
    work,
    lwork,
    devInfo,
    params));
    
    mxGPUDestroyGPUArray(denseReconstructed);
    
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>(min(numARows, numAColumns)+1)){
        printf("This function dose not converge under given tolerance and maximum sweeps:");
 
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
   
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
     
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 
    cusolverDnDestroyGesvdjInfo(params);
	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
    }        
  
    
    mxDestroyArray(row_sort);
    mxDestroyArray(col_sort);
    mxDestroyArray(tempx); 
	cusparseDestroyMatDescr(descrA);  
	cusparseDestroy(handle);     
 
    } 
    
   else{
    
   const mwSize *dimsA;
   dimsA=mxGPUGetDimensions(INPUTMATRIXGPU);
   numARows = (int)dimsA[0]; /* gets number of rows of A */
   numAColumns = (int)dimsA[1]; /* gets number of columns of A */
   
  if ( numARows < numAColumns ) {
       
         mxGPUDestroyGPUArray(INPUTMATRIXGPU);   
       
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,(first) argument must be a sparse/dense tall (numARows > numAColumns) or square matrix.");
             
    } 
    // this uses QR algorithm (for large matrices)
    
  if ( numARows*numAColumns >= mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	//cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    double *d_A_dense;
    d_A_dense = (double *)(mxGPUGetDataReadOnly(INPUTMATRIXGPU));
        mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      
      cusolverSafeCall(cusolverDnDgesvd_bufferSize(cusolver_handle, numARows, numAColumns, &lwork));
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
         
      size_t pivot_dimensionsrwork[1] = {min(numARows, numAColumns)-1};
      mxGPUArray *RW = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrwork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *rwork = (double *)mxGPUGetData(RW); 
      
    cusolverSafeCall(cusolverDnDgesvd(cusolver_handle, 'A', 'A', numARows, numAColumns, d_A_dense, numARows,
    d_S, d_U, numARows, d_V, numAColumns, work, lwork, rwork, devInfo));
    
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>0){
        printf("The unconverged superdiagonal elements of an upper bidiagonal matrix:");
        
       mxArray *rt =mxCreateNumericMatrix((min(numARows, numAColumns)-1), 1, mxDOUBLE_CLASS, mxREAL);
       double *rwork_host = (double *)mxGetDoubles(rt);
       
        gpuErrchk(cudaMemcpy(rwork_host, rwork, (min(numARows, numAColumns)-1)*sizeof(double), cudaMemcpyDeviceToHost));
     for (int i = 0; i < (min(numARows, numAColumns)-1); i++){
        printf("rwork_host[%i] = %i \n", i, rwork_host[i]);    
         }
         printf("\n");
         mxDestroyArray(rt);
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 

	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
    }
   
   // this uses the parallelism of Jacobi method (for small and medium size matrices) 
  if ( numARows*numAColumns < mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	//cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    
    const double tol = 1.e-7;
    const int max_sweeps = 150;
    const int econ = 0 ; 

    
    double *d_A_dense;
    d_A_dense = (double *)(mxGPUGetDataReadOnly(INPUTMATRIXGPU));
        mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      gesvdjInfo_t params= NULL;
      
      
    cusolverSafeCall(cusolverDnCreateGesvdjInfo(&params));


    cusolverSafeCall(cusolverDnXgesvdjSetTolerance(
        params,
        tol));
    

    cusolverSafeCall(cusolverDnXgesvdjSetMaxSweeps(
        params,
        max_sweeps));
    
      
 
    cusolverSafeCall(cusolverDnDgesvdj_bufferSize(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,      
    numARows,             
    d_S, 
    d_U,      
    numARows,              
    d_V,      
    numAColumns,              
    &lwork,
    params));
      
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
      
    cusolverSafeCall(cusolverDnDgesvdj(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,            
    numARows,              
    d_S,       
    d_U,            
    numARows,              
    d_V,            
    numAColumns,              
    work,
    lwork,
    devInfo,
    params));
    
    
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>(min(numARows, numAColumns)+1)){
        printf("This function dose not converge under given tolerance and maximum sweeps:");
 
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
   
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
     
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 
    cusolverDnDestroyGesvdjInfo(params);
	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
    }

      }
      
   }
       
    else if (!(mxIsGPUArray(INPUTMATRIX))){
   
      //if (mxGetClassID(INPUTMATRIX) != mxDOUBLE_CLASS && (!mxIsComplex(prhs[0]))){
      //   mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
         //       "Invalid input to MEX file, input(FIRST ARGUMENT) must be  double precision.");
             
   // }
if(mxIsSparse(INPUTMATRIX)) {
    
      int numARows, numAColumns;
     numARows = (int)mxGetM(INPUTMATRIX); 
     numAColumns = (int)mxGetN(INPUTMATRIX);
  if ( numARows < numAColumns ) {
     
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,(first) argument must be a sparse/dense tall (numARows > numAColumns) or square matrix.");
             
    }
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
    double *d_A_dense = (double*)mxGPUGetData(denseReconstructed);
    	gpuErrchk(cudaMemcpy(d_A, h_A, nnz * sizeof(*d_A), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_ColIndices, h_A_ColIndices, (numAColumns + 1) * sizeof(*d_A_ColIndices), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_A_RowIndices, h_A_RowIndices, nnz *sizeof(*d_A_RowIndices), cudaMemcpyHostToDevice));

		cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSafeCall(cusparseDcsc2dense(handle, numARows, numAColumns, descrA, d_A, d_A_RowIndices, d_A_ColIndices,
			d_A_dense, numARows));
            
      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
      
      // this uses QR algorithm (for large matrices)       
      if ( numARows*numAColumns >= mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	//cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
   
      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      
      cusolverSafeCall(cusolverDnDgesvd_bufferSize(cusolver_handle, numARows, numAColumns, &lwork));
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
         
      size_t pivot_dimensionsrwork[1] = {min(numARows, numAColumns)-1};
      mxGPUArray *RW = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrwork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *rwork = (double *)mxGPUGetData(RW); 
      
    cusolverSafeCall(cusolverDnDgesvd(cusolver_handle, 'A', 'A', numARows, numAColumns, d_A_dense, numARows,
    d_S, d_U, numARows, d_V, numAColumns, work, lwork, rwork, devInfo));
    
        mxGPUDestroyGPUArray(denseReconstructed);
    
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>0){
        printf("The unconverged superdiagonal elements of an upper bidiagonal matrix:");
        
       mxArray *rt =mxCreateNumericMatrix((min(numARows, numAColumns)-1), 1, mxDOUBLE_CLASS, mxREAL);
       double *rwork_host = (double *)mxGetDoubles(rt);
       
        gpuErrchk(cudaMemcpy(rwork_host, rwork, (min(numARows, numAColumns)-1)*sizeof(double), cudaMemcpyDeviceToHost));
     for (int i = 0; i < (min(numARows, numAColumns)-1); i++){
        printf("rwork_host[%i] = %i \n", i, rwork_host[i]);    
         }
         printf("\n");
         mxDestroyArray(rt);
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 

	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
    }
   
   // this uses the parallelism of Jacobi method (for small and medium size matrices) 
  if ( numARows*numAColumns < mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	//cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    
    const double tol = 1.e-7;
    const int max_sweeps = 150;
    const int econ = 0 ; 

      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      gesvdjInfo_t params= NULL;
      
      
    cusolverSafeCall(cusolverDnCreateGesvdjInfo(&params));


    cusolverSafeCall(cusolverDnXgesvdjSetTolerance(
        params,
        tol));
    

    cusolverSafeCall(cusolverDnXgesvdjSetMaxSweeps(
        params,
        max_sweeps));
    
      
 
    cusolverSafeCall(cusolverDnDgesvdj_bufferSize(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,      
    numARows,             
    d_S, 
    d_U,      
    numARows,              
    d_V,      
    numAColumns,              
    &lwork,
    params));
      
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
      
    cusolverSafeCall(cusolverDnDgesvdj(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,            
    numARows,              
    d_S,       
    d_U,            
    numARows,              
    d_V,            
    numAColumns,              
    work,
    lwork,
    devInfo,
    params));
    
        mxGPUDestroyGPUArray(denseReconstructed);
        
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>(min(numARows, numAColumns)+1)){
        printf("This function dose not converge under given tolerance and maximum sweeps:");
 
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
   
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
     
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 
    cusolverDnDestroyGesvdjInfo(params);
	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
    } 
       
		cusparseDestroyMatDescr(descrA);  
		cusparseDestroy(handle);
 

    }

    
else{    
          
     int numARows, numAColumns;
     numARows = (int)mxGetM(INPUTMATRIX); 
     numAColumns = (int)mxGetN(INPUTMATRIX);
    if ( numARows < numAColumns ) {
     
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,(first) argument must be a sparse/dense tall (numARows > numAColumns) or square matrix.");
             
    }
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    double *h_A_dense1;
    h_A_dense1 = (double *)mxGetDoubles(INPUTMATRIX);
   
      size_t pivot_dimensionsvalueDA[2] = {numARows, numAColumns};
      mxGPUArray *OUTMA = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsvalueDA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_A_dense = (double *)mxGPUGetData(OUTMA);
      gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
      // this uses QR algorithm (for large matrices)
    
  if ( numARows*numAColumns >= mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	
      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      
      cusolverSafeCall(cusolverDnDgesvd_bufferSize(cusolver_handle, numARows, numAColumns, &lwork));
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
         
      size_t pivot_dimensionsrwork[1] = {min(numARows, numAColumns)-1};
      mxGPUArray *RW = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrwork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *rwork = (double *)mxGPUGetData(RW); 
      
    cusolverSafeCall(cusolverDnDgesvd(cusolver_handle, 'A', 'A', numARows, numAColumns, d_A_dense, numARows,
    d_S, d_U, numARows, d_V, numAColumns, work, lwork, rwork, devInfo));
    
    mxGPUDestroyGPUArray(OUTMA);
        
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>0){
        printf("The unconverged superdiagonal elements of an upper bidiagonal matrix:");
        
       mxArray *rt =mxCreateNumericMatrix((min(numARows, numAColumns)-1), 1, mxDOUBLE_CLASS, mxREAL);
       double *rwork_host = (double *)mxGetDoubles(rt);
       
        gpuErrchk(cudaMemcpy(rwork_host, rwork, (min(numARows, numAColumns)-1)*sizeof(double), cudaMemcpyDeviceToHost));
     for (int i = 0; i < (min(numARows, numAColumns)-1); i++){
        printf("rwork_host[%i] = %i \n", i, rwork_host[i]);    
         }
         printf("\n");
         mxDestroyArray(rt);
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
    mxGPUDestroyGPUArray(RW);
    
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 

	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
    }
   
   // this uses the parallelism of Jacobi method (for small and medium size matrices) 
  if ( numARows*numAColumns < mediumsizematrix ) { 
  
	//cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusolverDnHandle_t cusolver_handle= NULL;
	cusolverDnCreate(&cusolver_handle);

	//cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    
    const double tol = 1.e-7;
    const int max_sweeps = 150;
    const int econ = 0 ; 

   
      size_t pivot_dimensionssingularvalue[1] = {min(numARows, numAColumns)};
      mxGPUArray *S = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionssingularvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_S = (double *)mxGPUGetData(S);
	
      size_t pivot_dimensionsleftunitarymatrix[2] = {numARows, numARows};
      mxGPUArray *U = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsleftunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_U = (double *)mxGPUGetData(U);
    
      size_t pivot_dimensionsrightunitarymatrix[2] = {numAColumns, numAColumns};
      mxGPUArray *V = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsrightunitarymatrix, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_V = (double *)mxGPUGetData(V);	
      
      int lwork = 0;
      size_t pivot_dimensionsInfo[1] = {1};
      mxGPUArray *In = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsInfo, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      int *devInfo = (int *)mxGPUGetData(In);
      gesvdjInfo_t params= NULL;
      
      
    cusolverSafeCall(cusolverDnCreateGesvdjInfo(&params));


    cusolverSafeCall(cusolverDnXgesvdjSetTolerance(
        params,
        tol));
    

    cusolverSafeCall(cusolverDnXgesvdjSetMaxSweeps(
        params,
        max_sweeps));
    
      
 
    cusolverSafeCall(cusolverDnDgesvdj_bufferSize(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,      
    numARows,             
    d_S, 
    d_U,      
    numARows,              
    d_V,      
    numAColumns,              
    &lwork,
    params));
      
      
      size_t pivot_dimensionswork[1] = {lwork};
      mxGPUArray *W = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionswork, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *work = (double *)mxGPUGetData(W);
      
      
    cusolverSafeCall(cusolverDnDgesvdj(
    cusolver_handle,
    CUSOLVER_EIG_MODE_VECTOR, 
    econ,             
    numARows,                
    numAColumns,                
    d_A_dense,            
    numARows,              
    d_S,       
    d_U,            
    numARows,              
    d_V,            
    numAColumns,              
    work,
    lwork,
    devInfo,
    params));
    
    mxGPUDestroyGPUArray(OUTMA);
        
    int devInfo_host = 0;  gpuErrchk(cudaMemcpy(&devInfo_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_host != 0){ 
    
    if (devInfo_host>(min(numARows, numAColumns)+1)){
        printf("This function dose not converge under given tolerance and maximum sweeps:");
 
    }
    
    if (devInfo_host<0){
        printf("the [%i]-th parameter is wrong \n", devInfo_host); 
    }
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
   
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);
    cusolverDnDestroy(cusolver_handle);
       
    mexErrMsgIdAndTxt( "MATLAB:mexatexit:fatal", "Unsuccessful SVD execution!");
        
        }
    
    
    mxGPUDestroyGPUArray(In);  
    mxGPUDestroyGPUArray(W); 
     
      LeftUnitaryMatrix = mxGPUCreateMxArrayOnGPU(U);
      SingularValueVector = mxGPUCreateMxArrayOnGPU(S);
      RightUnitaryMatrix = mxGPUCreateMxArrayOnGPU(V);
    
    mxGPUDestroyGPUArray(S);  
    mxGPUDestroyGPUArray(U); 
    mxGPUDestroyGPUArray(V);

    cusolverDnDestroy(cusolver_handle); 
    cusolverDnDestroyGesvdjInfo(params);
	//cusparseDestroyMatDescr(descrA);  
	//cusparseDestroy(handle);
    
        }  
  
     }
  }
        //
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
