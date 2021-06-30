
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


#include <cuda.h>
#include <cuda_runtime.h>
#include "SPARSEHELPER.h"
#include "ERRORCHK.h"

// Input Arguments
#define	INPUTMATRIX   prhs[0]


// Output Arguments
#define	ROW_SORT plhs[0]
#define	COL_SORT plhs[1]
#define	VAL_SORT plhs[2]


                     
    
extern "C" static void mexCuMatlab_findD(int nlhs, mxArray *plhs[],
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
       
   // if (mxGPUGetClassID(INPUTMATRIXGPU) != mxDOUBLE_CLASS && mxGPUGetComplexity(INPUTMATRIXGPU) == mxREAL ) {
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
            //    "Invalid input to MEX file, input(FIRST ARGUMENT) must be  double precision.");
   // }
       
 
       
    const mwSize *dimsGPU;
    dimsGPU=mxGPUGetDimensions(INPUTMATRIXGPU);
    
    numARows = (int)dimsGPU[0]; /* gets number of rows of A */
    numAColumns = (int)dimsGPU[1]; /* gets number of columns of A */

     mwIndex nnz1;
    
    mxArray * tempx = mxGPUCreateMxArrayOnCPU(INPUTMATRIXGPU);
    nnz1 = *(mxGetJc(tempx) + numAColumns);
     //nnz1=(mwSize)ceil(numARows*numAColumns);
    int nnz= (int)nnz1;


    mxArray *row_sort =mxCreateNumericMatrix(nnz, 1, mxDOUBLE_CLASS, mxREAL);
    double *pointerrow = (double *)mxGetDoubles(row_sort);
   
    Ir_DataGetSetDXY(tempx , pointerrow, nnz);

    mxArray *col_sort =mxCreateNumericMatrix(nnz, 1, mxDOUBLE_CLASS, mxREAL);
    double *pointercol = (double *)mxGetDoubles(col_sort);
    
    Jc_GetSetDXY(tempx , pointercol);
   
    double  *pointerval = (double *)mxGetDoubles(tempx);
            
   size_t pivot_dimensionsrow[1] = {nnz};
   size_t pivot_dimensionscolumn[1] = {nnz}; 
   size_t pivot_dimensionsvalue[1] = {nnz};    
      mxGPUArray *row_sortC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        double *xrow_sortC=(double*)mxGPUGetData(row_sortC);
 gpuErrchk(cudaMemcpy(xrow_sortC, pointerrow, nnz * sizeof(*xrow_sortC), cudaMemcpyHostToDevice));
      mxGPUArray *col_sortC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        double *xcol_sortC=(double*)mxGPUGetData(col_sortC);
 gpuErrchk(cudaMemcpy(xcol_sortC, pointercol, nnz * sizeof(*xcol_sortC), cudaMemcpyHostToDevice));       
      mxGPUArray *val_sortC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        double *xval_sortC=(double*)mxGPUGetData(val_sortC);
 gpuErrchk(cudaMemcpy(xval_sortC, pointerval, nnz * sizeof(*xval_sortC), cudaMemcpyHostToDevice)); 
            
      ROW_SORT = mxGPUCreateMxArrayOnGPU(row_sortC);
      COL_SORT = mxGPUCreateMxArrayOnGPU(col_sortC);
      VAL_SORT = mxGPUCreateMxArrayOnGPU(val_sortC);       
     
      mxGPUDestroyGPUArray(row_sortC); 
      mxGPUDestroyGPUArray(col_sortC);
      mxGPUDestroyGPUArray(val_sortC);    
      mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      mxDestroyArray(tempx);
      mxDestroyArray(row_sort);
      mxDestroyArray(col_sort);
 
    } 
    
   else{
    
   const mwSize *dimsA;
   dimsA=mxGPUGetDimensions(INPUTMATRIXGPU);
   numARows = (int)dimsA[0]; /* gets number of rows of A */
   numAColumns = (int)dimsA[1]; /* gets number of columns of A */
    
   //if (mxGPUGetClassID(INPUTMATRIXGPU) != mxDOUBLE_CLASS) {
       //  mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
            //    "Invalid input to MEX file, input(FIRST ARGUMENT) must be double precision.");
   // }
   
	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    double *d_A_dense;
    d_A_dense = (double *)(mxGPUGetDataReadOnly(INPUTMATRIXGPU));
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

            
      ROW_SORT = mxGPUCreateMxArrayOnGPU(row_sort);
      COL_SORT = mxGPUCreateMxArrayOnGPU(col_sort);
      VAL_SORT = mxGPUCreateMxArrayOnGPU(val_sort);
      
          
    
    gpuErrchk(cudaFree(pBuffer));
	gpuErrchk(cudaFree(P));
      mxGPUDestroyGPUArray(PerVect);
      mxGPUDestroyGPUArray(RowIndi);
      mxGPUDestroyGPUArray(row_sort);
      mxGPUDestroyGPUArray(col_sort);
      mxGPUDestroyGPUArray(val_sort);
      mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      

	cusparseDestroyMatDescr(descrA);  
	cusparseDestroy(handle);
    

    }
   }
       
    else if (!(mxIsGPUArray(INPUTMATRIX))){
   
      //if (mxGetClassID(INPUTMATRIX) != mxDOUBLE_CLASS && (!mxIsComplex(prhs[0]))){
      //   mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
         //       "Invalid input to MEX file, input(FIRST ARGUMENT) must be  double precision.");
             
   // }
if(mxIsSparse(INPUTMATRIX)) {
    
 
    const mwSize *dimsCPU;
    dimsCPU=mxGetDimensions(INPUTMATRIX);
    
   // int  numARows = (int)dimsCPU[0]; /* gets number of rows of A */
    int  numAColumns = (int)dimsCPU[1]; /* gets number of columns of A */

    mwIndex nnz1;
    
    nnz1 = *(mxGetJc(INPUTMATRIX) + numAColumns);
    int nnz= (int)nnz1;
   


    //const mwSize ndim = 1;
   // mwSize dims[ndim];

   // dims[0] = nnz;

    ROW_SORT = mxCreateNumericMatrix(nnz, 1, mxDOUBLE_CLASS, mxREAL);
    double  *ROWSORT  = (double *)mxGetDoubles(ROW_SORT);
           
        Ir_DataDX(INPUTMATRIX, ROWSORT);   

    COL_SORT = mxCreateNumericMatrix(nnz, 1, mxDOUBLE_CLASS, mxREAL);
    double  *COLSORT = (double *)mxGetDoubles(COL_SORT);
 
      Jc_SetDX(INPUTMATRIX, COLSORT);
      
    VAL_SORT = mxCreateNumericMatrix(nnz, 1, mxDOUBLE_CLASS, mxREAL);
    double  *VALSORT = (double *)mxGetDoubles(VAL_SORT);
    value_DataGetSetDXY(INPUTMATRIX, VALSORT,nnz ); 

    }

    
else{    
          
     int numARows, numAColumns;
     numARows = (int)mxGetM(INPUTMATRIX); 
     numAColumns = (int)mxGetN(INPUTMATRIX);

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
	 
   	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    double *h_A_dense1;
    h_A_dense1 = (double *)mxGetDoubles(INPUTMATRIX);
    
   // double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
    
     size_t pivot_dimensionsvalueDA[2] = {numARows, numAColumns};
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
        //gpuErrchk(cudaFree(d_A_dense));
        mxGPUDestroyGPUArray(OUTMA);
        
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

            
      ROW_SORT = mxGPUCreateMxArrayOnGPU(row_sort);
      COL_SORT = mxGPUCreateMxArrayOnGPU(col_sort);
      VAL_SORT = mxGPUCreateMxArrayOnGPU(val_sort);
      
          
    
      gpuErrchk(cudaFree(pBuffer));
	  gpuErrchk(cudaFree(P));
      mxGPUDestroyGPUArray(PerVect);
      mxGPUDestroyGPUArray(RowIndi);
      mxGPUDestroyGPUArray(row_sort);
      mxGPUDestroyGPUArray(col_sort);
      mxGPUDestroyGPUArray(val_sort);
     // mxGPUDestroyGPUArray(INPUTMATRIXGPU);
      

	cusparseDestroyMatDescr(descrA);   
	cusparseDestroy(handle);
        
        }
      }
        //
    else{
        mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Incorrect input arguments! %s\n");    
        }

}
