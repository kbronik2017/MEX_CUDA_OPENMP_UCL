
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0],prhs[1],prhs[2]  := mxGPUArray or CPU Array}[double or complex double]
 * Create sparse matrix  
 * Z=CuMatlab_sparse(X) 
 * Z=CuMatlab_sparse(X,Y)
 * Z=CuMatlab_sparse(X,Y,Z)
 * Z=CuMatlab_sparse(X,Y,Z,row,column) 
 * Z=CuMatlab_sparse(X,Y,Z,row,column,nz)
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
#define	INPUTMATRIX   prhs[0]


// Output Arguments
#define	OUTPUTMATRIX   plhs[0]


    
extern "C" static void mexCuMatlab_sparseO(int nlhs, mxArray *plhs[],
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
 input_buf0 = mxArrayToString(prhs[0]);

      if ((mxIsChar(prhs[0]))){
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Input(FIRST ARGUMENT) must be array, or gpuArray object not  %s\n",input_buf0);
    }
    
    
    
    


if (mxIsGPUArray(prhs[0])) {
		
    
    mxGPUArray const *INPUTMATRIXGPUx;
     
    mxInitGPU();
	// mxGPUIsSparse(INPUTMATRIXGPUx)==1
    INPUTMATRIXGPUx = mxGPUCreateFromMxArray(INPUTMATRIX);
   //if(mxIsSparse(mxGPUCreateMxArrayOnCPU(INPUTMATRIXGPUx))) {
     //  plhs[0] = mxDuplicateArray((mxGPUCreateMxArrayOnCPU(INPUTMATRIXGPUx)));
 if(mxGPUIsSparse(INPUTMATRIXGPUx)==1) {
       plhs[0] = mxGPUCreateMxArrayOnGPU(INPUTMATRIXGPUx);
       printf("Warning! Input(FIRST ARGUMENT) must be non sparse!, continuing execution... \n");  
       mxGPUDestroyGPUArray(INPUTMATRIXGPUx);
                return;
    
    } 
    
   else{
    
   if (mxGPUGetClassID(INPUTMATRIXGPUx) != mxDOUBLE_CLASS) {
         mxGPUDestroyGPUArray(INPUTMATRIXGPUx);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be  double precision.");
    }
   //if ( mxGPUGetComplexity(INPUTMATRIXGPUx) != mxREAL) {
      //   mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
         //       "Invalid input to MEX file, input(FIRST ARGUMENT) must be real with no imaginary components.");
 //   }
    
    
    if (mxGPUGetClassID(INPUTMATRIXGPUx) == mxDOUBLE_CLASS && mxGPUGetComplexity(INPUTMATRIXGPUx) == mxREAL){
     //mxGPUArray  *INPUTMATRIXGPU;
     //INPUTMATRIXGPU= mxGPUCreateFromMxArray(INPUTMATRIX);  
    const mwSize *dimsGPU;
    dimsGPU=mxGPUGetDimensions(INPUTMATRIXGPUx);
    int numARows, numAColumns;
    numARows = (int)dimsGPU[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPU[1]; /* gets number of columns of A */

      double const *d_A_dense = (double const *) mxGPUGetDataReadOnly(INPUTMATRIXGPUx);
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
   // int *h_nnzPerVector = (int *)malloc(numARows * sizeof(*h_nnzPerVector));
    //int nnz = NUMZ(hostA, numARows, numAColumns, h_nnzPerVector);
   // int nnz =NUMCSCGPOO(d_host, numARows, numAColumns, h_nnzPerVector);
     		int nnz = 0;                            // --- Number of nonzero elements in dense matrix
		const int lda = numARows;
		//int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, numAColumns * sizeof(*d_nnzPerVector)));
		
	size_t pivot_pervect1[1] = {numAColumns};
    mxGPUArray *PerVect1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect1, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVector = (int*)mxGPUGetData(PerVect1);	
		
		
		cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_COLUMN, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
	
   size_t pivot_dimensionsrow[1] = {nnz};
   size_t pivot_dimensionscolumn[1] = {numAColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnz};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORT = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORT = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *VALSORT = (double *)mxGPUGetData(VAL_SORT1);
   mwSize nnzm=(mwSize)nnz;
   OUTPUTMATRIX = mxCreateSparse(numARows,numAColumns,nnzm,mxREAL);
    
cusparseSafeCall(cusparseDdense2csc(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, VALSORT, ROWSORT, COLSORT));    
     
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORT1);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORT1);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  
        irs = static_cast<mwIndex *> (mxMalloc (nnz * sizeof(mwIndex)));
       int i;
	   #pragma omp parallel for shared(nnz) private(i)
         for (i = 0; i < nnz; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i]);  
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((numAColumns+1) * sizeof(mwIndex)));
      int nc1= numAColumns+1;
      #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i]);
            }
             
                   
        mxDouble* PRS = (mxDouble*) mxMalloc (nnz * sizeof(mxDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORT, nnz * sizeof(mxDouble), cudaMemcpyDeviceToHost))

   
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles (OUTPUTMATRIX)) ;
    
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
       // mxSetPr(OUTPUTMATRIX, (double *)PRS);
        int s = mxSetDoubles(OUTPUTMATRIX, (mxDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
         mxDestroyArray(RS);
         mxDestroyArray(CS);
 
      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
      mxGPUDestroyGPUArray(INPUTMATRIXGPUx);
     // mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      mxGPUDestroyGPUArray(PerVect1); 
      cusparseDestroyMatDescr(descrA);     
       cusparseDestroy(handle); 
    }
    
   if (mxGPUGetClassID(INPUTMATRIXGPUx) == mxDOUBLE_CLASS && mxGPUGetComplexity(INPUTMATRIXGPUx) == mxCOMPLEX){
  
       //mxGPUArray  *INPUTMATRIXGPU;
     //INPUTMATRIXGPU= mxGPUCreateFromMxArray(INPUTMATRIX);  
    const mwSize *dimsGPU;
    dimsGPU=mxGPUGetDimensions(INPUTMATRIXGPUx);
    int numARows, numAColumns;
    numARows = (int)dimsGPU[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPU[1]; /* gets number of columns of A */

      cuDoubleComplex const *d_A_dense = (cuDoubleComplex const *) mxGPUGetDataReadOnly(INPUTMATRIXGPUx);
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
   // int *h_nnzPerVector = (int *)malloc(numARows * sizeof(*h_nnzPerVector));
    //int nnz = NUMZ(hostA, numARows, numAColumns, h_nnzPerVector);
   // int nnz =NUMCSCGPOO(d_host, numARows, numAColumns, h_nnzPerVector);
     		int nnz = 0;                            // --- Number of nonzero elements in dense matrix
		const int lda = numARows;
		//int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, numAColumns * sizeof(*d_nnzPerVector)));
		
    size_t pivot_pervect1[1] = {numAColumns};
    mxGPUArray *PerVect1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect1, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVector = (int*)mxGPUGetData(PerVect1);	
		
		cusparseSafeCall(cusparseZnnz(handle, CUSPARSE_DIRECTION_COLUMN, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
	
   size_t pivot_dimensionsrow[1] = {nnz};
   size_t pivot_dimensionscolumn[1] = {numAColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnz};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORT = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORT = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *VALSORT = (cuDoubleComplex *)mxGPUGetData(VAL_SORT1);
   mwSize nnzm=(mwSize)nnz;
   OUTPUTMATRIX = mxCreateSparse(numARows,numAColumns,nnzm,mxCOMPLEX);
    
cusparseSafeCall(cusparseZdense2csc(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, VALSORT, ROWSORT, COLSORT));    
     
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORT1);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORT1);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  
        irs = static_cast<mwIndex *> (mxMalloc (nnz * sizeof(mwIndex)));
       int i;
	   #pragma omp parallel for shared(nnz) private(i)
         for (i = 0; i < nnz; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i]);  
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((numAColumns+1) * sizeof(mwIndex)));
      int nc1= numAColumns+1;
      #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i]);
            }
             
        mxComplexDouble* PRS = (mxComplexDouble*) mxMalloc (nnz * sizeof(mxComplexDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORT, nnz * sizeof(mxComplexDouble), cudaMemcpyDeviceToHost));
  
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles (OUTPUTMATRIX)) ;
    
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
       // mxSetPr(OUTPUTMATRIX, (double *)PRS);
        int s = mxSetComplexDoubles(OUTPUTMATRIX, (mxComplexDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
         mxDestroyArray(RS);
         mxDestroyArray(CS);

      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
      mxGPUDestroyGPUArray(INPUTMATRIXGPUx);
     // mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      mxGPUDestroyGPUArray(PerVect1);  
      cusparseDestroyMatDescr(descrA);	  
       cusparseDestroy(handle); 
       
      
       }
     }
   }
     
////////////////////////////////////////////////////////////////////////////////////  
    else if (!(mxIsGPUArray(INPUTMATRIX))){
   
      if (mxGetClassID(INPUTMATRIX) != mxDOUBLE_CLASS ) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, input(FIRST ARGUMENT) must be double precision.");
             
    }
   //if ( mxIsComplex(INPUTMATRIX)) {
      //   mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
          //      "Invalid input to MEX file, input(FIRST ARGUMENT) must be real with no imaginary components.");
   // } 
if(mxIsSparse(INPUTMATRIX)) {
    
    plhs[0] = mxDuplicateArray(INPUTMATRIX);
       printf("Warning! Input(FIRST ARGUMENT) must be non sparse!, continuing execution... \n");   
                return;
        
    }
    
else{    
    if (mxGetClassID(INPUTMATRIX) == mxDOUBLE_CLASS  && (!mxIsComplex(INPUTMATRIX))){  

     int numARows, numAColumns;
     numARows = (int)mxGetM(INPUTMATRIX); 
     numAColumns = (int)mxGetN(INPUTMATRIX);

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
		
    double  * hostA ; // The A matrix
	hostA = (double *)mxGetDoubles(INPUTMATRIX);
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    
		//double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
	  size_t pivot_dimensionsvalueD[2] = {numARows, numAColumns};
      mxGPUArray *OUTM = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsvalueD, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double  *d_A_dense = (double *)mxGPUGetData(OUTM);	
		
		
		gpuErrchk(cudaMemcpy(d_A_dense, hostA, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

		int nnz = 0;                            // --- Number of nonzero elements in dense matrix
		const int lda = numARows;
		//int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, numAColumns * sizeof(*d_nnzPerVector)));
		
	size_t pivot_pervect1[1] = {numAColumns};
    mxGPUArray *PerVect1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect1, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVector = (int*)mxGPUGetData(PerVect1);		
		
		cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_COLUMN, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
	
   size_t pivot_dimensionsrow[1] = {nnz};
   size_t pivot_dimensionscolumn[1] = {numAColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnz};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORT = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORT = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *VALSORT = (double *)mxGPUGetData(VAL_SORT1);
   mwSize nnzm=(mwSize)nnz;
   OUTPUTMATRIX = mxCreateSparse(numARows,numAColumns,nnzm,mxREAL);
    
cusparseSafeCall(cusparseDdense2csc(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, VALSORT, ROWSORT, COLSORT));    
     
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORT1);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORT1);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  
        irs = static_cast<mwIndex *> (mxMalloc (nnz * sizeof(mwIndex)));
       int i;
	   #pragma omp parallel for shared(nnz) private(i)
         for (i = 0; i < nnz; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i]);  
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((numAColumns+1) * sizeof(mwIndex)));
      int nc1= numAColumns+1;
      #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i]);
            }
             
        mxDouble* PRS = (mxDouble*) mxMalloc (nnz * sizeof(mxDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORT, nnz * sizeof(mxDouble), cudaMemcpyDeviceToHost))
        
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles (OUTPUTMATRIX)) ;
    
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
       // mxSetPr(OUTPUTMATRIX, (double *)PRS);
        int s = mxSetDoubles(OUTPUTMATRIX, (mxDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
         mxDestroyArray(RS);
         mxDestroyArray(CS);
 
      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
      mxGPUDestroyGPUArray(OUTM);
     // mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      mxGPUDestroyGPUArray(PerVect1); 
      cusparseDestroyMatDescr(descrA);    
       cusparseDestroy(handle); 
   
        }
     
    if (mxGetClassID(INPUTMATRIX) == mxDOUBLE_CLASS  && (mxIsComplex(INPUTMATRIX))){  

     int numARows, numAColumns;
     numARows = (int)mxGetM(INPUTMATRIX); 
     numAColumns = (int)mxGetN(INPUTMATRIX);

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
		
    cuDoubleComplex  * hostA ; // The A matrix
	hostA = (cuDoubleComplex *)mxGetComplexDoubles(INPUTMATRIX);
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    //cuDoubleComplex *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns  * sizeof(*d_A_dense)));
	
	  size_t pivot_dimensionsvalueDC[2] = {numARows, numAColumns};
      mxGPUArray *OUTM = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsvalueDC, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
      cuDoubleComplex *d_A_dense = (cuDoubleComplex *)mxGPUGetData(OUTM);
	  
	  
		gpuErrchk(cudaMemcpy(d_A_dense, hostA, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

		int nnz = 0;                            // --- Number of nonzero elements in dense matrix
		const int lda = numARows;
		//int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, numAColumns * sizeof(*d_nnzPerVector)));
	size_t pivot_pervect1[1] = {numAColumns};
    mxGPUArray *PerVect1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect1, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVector = (int*)mxGPUGetData(PerVect1);		
		
		
		cusparseSafeCall(cusparseZnnz(handle, CUSPARSE_DIRECTION_COLUMN, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
	
   size_t pivot_dimensionsrow[1] = {nnz};
   size_t pivot_dimensionscolumn[1] = {numAColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnz};
   mxGPUArray * ROW_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORT = (int *)mxGPUGetData(ROW_SORT1);
   mxGPUArray * COL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORT = (int *)mxGPUGetData(COL_SORT1);
    mxGPUArray *VAL_SORT1 = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuDoubleComplex  *VALSORT = (cuDoubleComplex *)mxGPUGetData(VAL_SORT1);
   mwSize nnzm=(mwSize)nnz;
   OUTPUTMATRIX = mxCreateSparse(numARows,numAColumns,nnzm,mxCOMPLEX);
    
cusparseSafeCall(cusparseZdense2csc(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVector, VALSORT, ROWSORT, COLSORT));    
     
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORT1);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORT1);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  
        irs = static_cast<mwIndex *> (mxMalloc (nnz * sizeof(mwIndex)));
       int i;
	   #pragma omp parallel for shared(nnz) private(i)
         for (i = 0; i < nnz; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i]);  
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((numAColumns+1) * sizeof(mwIndex)));
      int nc1= numAColumns+1;
      #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i]);
            }
             
        mxComplexDouble* PRS = (mxComplexDouble*) mxMalloc (nnz * sizeof(mxComplexDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORT, nnz * sizeof(mxComplexDouble), cudaMemcpyDeviceToHost));
            
   
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles (OUTPUTMATRIX)) ;
    
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
       // mxSetPr(OUTPUTMATRIX, (double *)PRS);
        int s = mxSetComplexDoubles(OUTPUTMATRIX, (mxComplexDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
         mxDestroyArray(RS);
         mxDestroyArray(CS);

      mxGPUDestroyGPUArray(ROW_SORT1);
      mxGPUDestroyGPUArray(COL_SORT1);
      mxGPUDestroyGPUArray(VAL_SORT1);
      mxGPUDestroyGPUArray(OUTM);
     // mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      mxGPUDestroyGPUArray(PerVect1); 
      cusparseDestroyMatDescr(descrA);    
       cusparseDestroy(handle); 
        

        }
        
     }
        
               }
        //
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
