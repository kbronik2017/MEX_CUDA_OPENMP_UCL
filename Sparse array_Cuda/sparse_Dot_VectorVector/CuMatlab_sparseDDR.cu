
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
 * Sparse/Dense vector-sparse/dense vector dot product  Z=CuMatlab_dot(Sparse/Dense(X),Sparse/Dense(Y)).
 * Z= X.Y
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
    if ( (((numARows!= 1) && (numAColumns!= 1))) ||(((numBRows!= 1) && (numBColumns!= 1)))) {
              mxGPUDestroyGPUArray(INPUTDENSEGPUB);
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, first/second arguments must be dense/dense column/row vectors");
             
    }
    
    if ( mxGPUGetNumberOfElements(INPUTDENSEGPUA)!=mxGPUGetNumberOfElements(INPUTDENSEGPUB)) {
              mxGPUDestroyGPUArray(INPUTDENSEGPUB);
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, row/column number of dense vector (first argument) must be equal to row/column numbers of dense vector(second argument).");
             
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

    size_t pivot_pervect[1] = {numARows};
    mxGPUArray *PerVect = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect);
	
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));
    
    size_t pivot_rowindi[1] = {numARows + 1};
    mxGPUArray *RowIndi = mxGPUCreateGPUArray(1, (mwSize*) pivot_rowindi, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_A_RowIndices = (int*)mxGPUGetData(RowIndi);
		
   size_t pivot_dimensionsrow[1] = {nnzA};
   size_t pivot_dimensionscolumn[1] = {nnzA}; 
   size_t pivot_dimensionsvalue[1] = {nnzA};
    mxGPUArray *row_sort = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    
    mxGPUArray *col_sort = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    
    mxGPUArray *val_sort = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int *d_col_sort = (int*)mxGPUGetData(col_sort);
    int *d_row_sort = (int*)mxGPUGetData(row_sort);
    double *d_value_sort = (double*)mxGPUGetData(val_sort);
		cusparseSafeCall(cusparseDdense2csr(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, d_value_sort, d_A_RowIndices, d_col_sort));
       // gpuErrchk(cudaFree(d_nnzPerVectorA));
              mxGPUDestroyGPUArray(PerVect);
        
        cusparseSafeCall(cusparseXcsr2coo(handle,
		d_A_RowIndices,
		nnzA,
		numARows,
		d_row_sort,
		CUSPARSE_INDEX_BASE_ONE));
//gpuErrchk(cudaFree(d_A_RowIndices));
	// Sort by rows
	int *P = NULL;
	void *pBuffer = NULL;
	size_t pBufferSizeInBytes = 0;
	cusparseXcoosort_bufferSizeExt(handle, numARows, numAColumns,
		nnzA,
		d_row_sort,
		d_col_sort, &pBufferSizeInBytes);
	 
	gpuErrchk(cudaMalloc(&pBuffer, sizeof(char)*pBufferSizeInBytes));
	gpuErrchk(cudaMalloc(&P, sizeof(int)*nnzA));
	cusparseCreateIdentityPermutation(handle, nnzA, P);  

	cusparseSafeCall(cusparseXcoosortByColumn(handle, numAColumns, numAColumns,
		nnzA,
		d_row_sort,
		d_col_sort,
		P,
		pBuffer));
	
	cusparseSafeCall(cusparseDgthr(handle, nnzA, d_value_sort, d_value_sort, P, CUSPARSE_INDEX_BASE_ZERO));	
  

      gpuErrchk(cudaFree(pBuffer));
	  gpuErrchk(cudaFree(P));

      mxGPUDestroyGPUArray(RowIndi);
      
      
   double const *d_B_dense;
   d_B_dense = (double const *)(mxGPUGetDataReadOnly(INPUTDENSEGPUB));   
             mxGPUDestroyGPUArray(INPUTDENSEGPUB);
   
   double VALOUT=0; 
  if (numAColumns == 1) { 
  cusparseSafeCall(cusparseDdoti(handle, nnzA, 
              d_value_sort, 
              d_row_sort, d_B_dense, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));              
 
  }
 if (numARows == 1) {
  cusparseSafeCall(cusparseDdoti(handle, nnzA, 
              d_value_sort, 
              d_col_sort, d_B_dense, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));

 }

  OUTPUTMATRIX = mxCreateDoubleMatrix(1, 1, mxREAL);
 //*mxGetPr(OUTPUTMATRIX) = VALOUT;
 *mxGetDoubles(OUTPUTMATRIX) = static_cast<mxDouble> (VALOUT);
 
        mxGPUDestroyGPUArray(row_sort);
        mxGPUDestroyGPUArray(val_sort);
        mxGPUDestroyGPUArray(col_sort);
        cusparseDestroyMatDescr(descrA);   
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
    
    if ( (((numARows!= 1) && (numAColumns!= 1))) ||(((numBRows!= 1) && (numBColumns!= 1)))) {
    
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                       "Invalid input to MEX file, first/second arguments must be dense/dense column/row vectors");
             
    }
    
    if ( mxGetNumberOfElements(INPUTDENSEA)!=mxGetNumberOfElements(INPUTDENSEB)) {
      
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, row/column number of dense vector (first argument) must be equal to row/column numbers of dense vector(second argument).");
             
    }
      
      
   double *h_A_dense1;
   h_A_dense1 = (double *)mxGetDoubles(INPUTDENSEA);
   
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    
 
   size_t pivot_dimensionsvalueDA[2] = {numARows,numAColumns};
      mxGPUArray *OUTMA = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsvalueDA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_A_dense = (double *)mxGPUGetData(OUTMA);
	  
	  
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
    
    int nnzA = 0;                            // --- Number of nonzero elements in dense matrix A
	const int lda = numARows;

    size_t pivot_pervect[1] = {numARows};
    mxGPUArray *PerVect = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect);
	
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));
    
    size_t pivot_rowindi[1] = {numARows + 1};
    mxGPUArray *RowIndi = mxGPUCreateGPUArray(1, (mwSize*) pivot_rowindi, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_A_RowIndices = (int*)mxGPUGetData(RowIndi);
		
   size_t pivot_dimensionsrow[1] = {nnzA};
   size_t pivot_dimensionscolumn[1] = {nnzA}; 
   size_t pivot_dimensionsvalue[1] = {nnzA};
    mxGPUArray *row_sort = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    
    mxGPUArray *col_sort = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    
    mxGPUArray *val_sort = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int *d_col_sort = (int*)mxGPUGetData(col_sort);
    int *d_row_sort = (int*)mxGPUGetData(row_sort);
    double *d_value_sort = (double*)mxGPUGetData(val_sort);
		cusparseSafeCall(cusparseDdense2csr(handle, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, d_value_sort, d_A_RowIndices, d_col_sort));
       // gpuErrchk(cudaFree(d_nnzPerVectorA));
            mxGPUDestroyGPUArray(OUTMA);  
            mxGPUDestroyGPUArray(PerVect);
        cusparseSafeCall(cusparseXcsr2coo(handle,
		d_A_RowIndices,
		nnzA,
		numARows,
		d_row_sort,
		CUSPARSE_INDEX_BASE_ONE));
//gpuErrchk(cudaFree(d_A_RowIndices));
	// Sort by rows
	int *P = NULL;
	void *pBuffer = NULL;
	size_t pBufferSizeInBytes = 0;
	cusparseXcoosort_bufferSizeExt(handle, numARows, numAColumns,
		nnzA,
		d_row_sort,
		d_col_sort, &pBufferSizeInBytes);
	 
	gpuErrchk(cudaMalloc(&pBuffer, sizeof(char)*pBufferSizeInBytes));
	gpuErrchk(cudaMalloc(&P, sizeof(int)*nnzA));
	cusparseCreateIdentityPermutation(handle, nnzA, P);  

	cusparseSafeCall(cusparseXcoosortByColumn(handle, numAColumns, numAColumns,
		nnzA,
		d_row_sort,
		d_col_sort,
		P,
		pBuffer));
	
	cusparseSafeCall(cusparseDgthr(handle, nnzA, d_value_sort, d_value_sort, P, CUSPARSE_INDEX_BASE_ZERO));	
 
      gpuErrchk(cudaFree(pBuffer));
	  gpuErrchk(cudaFree(P));
      
      mxGPUDestroyGPUArray(RowIndi);

   
   double *h_B_dense1;
   h_B_dense1 = (double *)mxGetDoubles(INPUTDENSEB);    
     	
   double  *VALDENSE=0;
   mxGPUArray *VAL;
if (numBColumns == 1) {	
    
  size_t   pivot_dimensionsvalueV[1] = {numBRows};
  VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    VALDENSE = (double *)mxGPUGetData(VAL);
       gpuErrchk(cudaMemcpy(VALDENSE, h_B_dense1, sizeof(double) * numBRows , cudaMemcpyHostToDevice));
    }  

if (numBRows == 1) {
	
   size_t  pivot_dimensionsvalueV[1] = {numBColumns};
   VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    VALDENSE = (double *)mxGPUGetData(VAL);
       gpuErrchk(cudaMemcpy(VALDENSE, h_B_dense1, sizeof(double) * numBColumns , cudaMemcpyHostToDevice));
   
}
      
    
   double VALOUT=0; 
    
  if (numAColumns == 1) { 
  cusparseSafeCall(cusparseDdoti(handle, nnzA, 
              d_value_sort, 
              d_row_sort, VALDENSE, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));              
 
  }
 if (numARows == 1) {
  cusparseSafeCall(cusparseDdoti(handle, nnzA, 
              d_value_sort, 
              d_col_sort, VALDENSE, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));

 }

			  
  OUTPUTMATRIX = mxCreateDoubleMatrix(1, 1, mxREAL);
 //*mxGetPr(OUTPUTMATRIX) = VALOUT;
 *mxGetDoubles(OUTPUTMATRIX) = static_cast<mxDouble> (VALOUT);
        mxGPUDestroyGPUArray(row_sort);
        mxGPUDestroyGPUArray(val_sort);
        mxGPUDestroyGPUArray(col_sort);		
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
