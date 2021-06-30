
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
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"
#include <omp.h>


// Input Arguments
#define	INPUTDENSEA   prhs[0]
#define	INPUTDENSEB   prhs[1]

// Output Arguments
#define	OUTPUTMATRIX  plhs[0]



  
    
extern "C" static void mexCuMatlab_sparseDDR(int nlhs, mxArray *plhs[],
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
 input_buf1 = mxArrayToString(INPUTDENSEB);

      if ((mxIsChar(INPUTDENSEB))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(SECOND ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf1);
    } 


if (mxIsGPUArray(INPUTDENSEA) && mxIsGPUArray(INPUTDENSEB)) {
    
    mxGPUArray const *INPUTDENSEGPUA;
    mxGPUArray const *INPUTDENSEGPUB;
    
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    INPUTDENSEGPUA = mxGPUCreateFromMxArray(INPUTDENSEA);
    INPUTDENSEGPUB = mxGPUCreateFromMxArray(INPUTDENSEB);
    
   
	
    if((!mxGPUIsSparse(INPUTDENSEGPUA))&& (!mxGPUIsSparse(INPUTDENSEGPUB)) ){
        
    const mwSize *dimsGPUSA;
    dimsGPUSA=mxGPUGetDimensions(INPUTDENSEGPUA);
    int numARows, numAColumns;
    numARows = (int)dimsGPUSA[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPUSA[1]; /* gets number of columns of A */
    
    const mwSize *dimsGPUSB;
    dimsGPUSB=mxGPUGetDimensions(INPUTDENSEGPUB);
    int numBRows, numBColumns;
    
    numBRows = (int)dimsGPUSB[0]; /* gets number of rows of B */
    numBColumns = (int)dimsGPUSB[1]; /* gets number of columns of B */

	    if ( numARows != numAColumns ) {
       
              mxGPUDestroyGPUArray(INPUTDENSEGPUB);
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);   
       
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file,first argument must be a sparse/dense square matrix.");
             
    } 
     if ( (numBColumns!= 1) ) {
         
              mxGPUDestroyGPUArray(INPUTDENSEGPUB);
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, second argument must be a dense/sparse column vector.");
             
    }
    if ( (numBRows!= numARows) ) {
              mxGPUDestroyGPUArray(INPUTDENSEGPUB);
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, array (matrix-vector) dimensions must agree.");
             
    }
	
	


  double const *d_A_dense;
   d_A_dense = (double const *)(mxGPUGetDataReadOnly(INPUTDENSEGPUA));
        mxGPUDestroyGPUArray(INPUTDENSEGPUA); 
      
	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);

	int nnzA = 0;                            // --- Number of nonzero elements in dense matrix A
	const int lda = numARows;
	//int *d_nnzPerVectorA;    //gpuErrchk(cudaMalloc(&d_nnzPerVectorA, numARows * sizeof(*d_nnzPerVectorA)));
    size_t pivot_pervect1[1] = {numARows};
    mxGPUArray *PerVect1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect1, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect1);
	//double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
	//gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));
        //double *d_A;            //gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
		//int *d_A_RowIndices;    //gpuErrchk(cudaMalloc(&d_A_RowIndices, (numARows + 1) * sizeof(*d_A_RowIndices)));
		//int *d_A_ColIndices;    //gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);

		cusparseSafeCall(cusparseDdense2csr(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));
             

        mxGPUDestroyGPUArray(PerVect1);
        
   double const *d_B_dense;
   d_B_dense = (double const *)(mxGPUGetDataReadOnly(INPUTDENSEGPUB));
   mxGPUDestroyGPUArray(INPUTDENSEGPUB);

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
    
    
    size_t pivot_dimensionsvalueV[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
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
    else if (!(mxIsGPUArray(INPUTDENSEA)) && !(mxIsGPUArray(INPUTDENSEB))){
   
     // if ((mxGetClassID(INPUTSPARSEA) != mxDOUBLE_CLASS) || (mxGetClassID(INPUTSPARSEB) != mxDOUBLE_CLASS)) {
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
           //     "Invalid input to MEX file, input(FIRST and SECOND  ARGUMENTS) must be  double precision.");
             
   // }
    if((!mxIsSparse(INPUTDENSEA))&& (!mxIsSparse(INPUTDENSEB)) ){
    
     mxInitGPU();
    const mwSize *dimsCPUA;
    dimsCPUA=mxGetDimensions(INPUTDENSEA);
    
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

    
   double *h_A_dense1;
   h_A_dense1 = (double *)mxGetDoubles(INPUTDENSEA);

      
	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);

	int nnzA = 0;                            // --- Number of nonzero elements in dense matrix A
	const int lda = numARows;
	//int *d_nnzPerVectorA;    gpuErrchk(cudaMalloc(&d_nnzPerVectorA, numARows * sizeof(*d_nnzPerVectorA)));
	size_t pivot_pervect1[1] = {numARows};
    mxGPUArray *PerVect1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect1, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect1);
	
	
	//double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
	
	 size_t pivot_dimensionsvalueDA[2] = {numARows, numAColumns};
      mxGPUArray *OUTMA = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsvalueDA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_A_dense = (double *)mxGPUGetData(OUTMA);
	  
	  
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));
        
    	//double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
		//int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (numARows + 1) * sizeof(*d_A_RowIndices)));
		//int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
   size_t pivot_dimensA[1] = {nnzA};
   size_t pivot_dimensROW_A[1] = {numARows+1};
   size_t pivot_dimensCOL_A[1] = {nnzA};
   
   
   mxGPUArray *A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_A = (double *)mxGPUGetData(A);
   mxGPUArray * ROW_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_RowIndices = (int *)mxGPUGetData(ROW_A);
   mxGPUArray * COL_A = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_A, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_A_ColIndices = (int *)mxGPUGetData(COL_A);

		cusparseSafeCall(cusparseDdense2csr(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));
		       
		mxGPUDestroyGPUArray(OUTMA);
		mxGPUDestroyGPUArray(PerVect1);
       
        
   double *h_B_dense1;
   h_B_dense1 = (double *)mxGetDoubles(INPUTDENSEB);    
        
 
	  size_t pivot_dimensionsvalueDB[1] = {numBRows};
      mxGPUArray *OUTMB = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueDB, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_B_dense = (double *)mxGPUGetData(OUTMB);
     gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense1, numBRows * sizeof(*d_B_dense), cudaMemcpyHostToDevice));
     
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
    
    size_t pivot_dimensionsvalueV[1] = {numAColumns};

    mxGPUArray *VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
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
