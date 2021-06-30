
/*
 * This CUDA-Cusparse code can handle/work with  any type of the input mxArrays, 
 * GPUarray or standard matlab CPU array as input {prhs[0]/prhs[1] := mxGPUArray or CPU Array}[double/complex double]
 * Sparse/Dense matrix-sparse/dense matrix multiplication   Z=CuMatlab_multiply(Sparse/Dense(X),Sparse/Dense(Y)).
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
#define	INPUTSPARSEB   prhs[1]

// Output Arguments
#define	OUTPUTMATRIX  plhs[0]



  
    
extern "C" static void mexCuMatlab_sparseDSR(int nlhs, mxArray *plhs[],
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
    if ( numAColumns != numBRows) {
		
		mxGPUDestroyGPUArray(INPUTDENSEGPUA);
        mxGPUDestroyGPUArray(INPUTSPARSEGPUB);
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, column number of dense matrix(first argument) must match the row number of sparse matrix(second argument).");
             
    }
   
   double const *d_A_dense;
   d_A_dense = (double const *)(mxGPUGetDataReadOnly(INPUTDENSEGPUA));
    
    mwIndex nnz2;
    mxArray * VLSXY2 = mxGPUCreateMxArrayOnCPU(INPUTSPARSEGPUB);
    nnz2 = *(mxGetJc(VLSXY2) + numBColumns);
    int nnzB = (int)nnz2;
          
    mxArray *  ROW_SORTB = mxCreateNumericMatrix(nnzB, 1,mxINT32_CLASS, mxREAL);
    int *ROWSORTB  = (int *)mxGetInt32s(ROW_SORTB);
       SetIr_Data(VLSXY2, ROWSORTB);
    
   mxArray *  COL_SORTB = mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    int  *COLSORTB = (int *)mxGetInt32s(COL_SORTB);
          SetJc_Int(VLSXY2, COLSORTB);
      
    double  *VALSORTB = (double *)mxGetDoubles(VLSXY2);
           
     cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);        
    
    
    int nnzA = 0;                            // --- Number of nonzero elements in dense matrix A
	const int lda = numARows;
	//int *d_nnzPerVectorA;   // gpuErrchk(cudaMalloc(&d_nnzPerVectorA, numARows * sizeof(*d_nnzPerVectorA)));
	
	size_t pivot_pervect[1] = {numARows};
    mxGPUArray *PerVect = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect);
	//double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
	//gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));      
       // double *d_A;           // gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
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
        //gpuErrchk(cudaFree(d_A_dense));
        mxGPUDestroyGPUArray(PerVect);
         mxGPUDestroyGPUArray(INPUTDENSEGPUA);
         mxGPUDestroyGPUArray(INPUTSPARSEGPUB);
	//double *d_B;            //gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));
	//int *d_B_RowIndices;    //gpuErrchk(cudaMalloc(&d_B_RowIndices, (numBRows + 1) * sizeof(*d_B_RowIndices)));
	//int *d_B_ColIndices;   // gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));
	//int *d_cooRowIndB;      // gpuErrchk(cudaMalloc(&d_cooRowIndB, nnzB * sizeof(*d_cooRowIndB)));
	
   size_t pivot_dimensB[1] = {nnzB};
   size_t pivot_dimensROW_B[1] = {numBRows+1};
   size_t pivot_dimensCOL_B[1] = {nnzB};
   size_t pivot_dimensCOO_B[1] = {nnzB};
   
   mxGPUArray *B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensB, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_B = (double *)mxGPUGetData(B);
   mxGPUArray * ROW_B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_B, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_B_RowIndices = (int *)mxGPUGetData(ROW_B);
   mxGPUArray * COL_B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_B, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_B_ColIndices = (int *)mxGPUGetData(COL_B);
    mxGPUArray * COO_B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOO_B, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_cooRowIndB = (int *)mxGPUGetData(COO_B);
	
	
	
	
	
	
	// --- Descriptor for sparse matrix B
	gpuErrchk(cudaMemcpy(d_B, VALSORTB, nnzB * sizeof(*d_B), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B_ColIndices, COLSORTB, nnzB * sizeof(*d_B_ColIndices), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cooRowIndB, ROWSORTB, nnzB * sizeof(*d_cooRowIndB), cudaMemcpyHostToDevice));
    
    
	int *Pb = NULL;
	void *pBufferb = NULL;
	size_t pBufferSizeInBytesb = 0;
	cusparseXcoosort_bufferSizeExt(handle, numBRows, numBColumns,
		nnzB,
		d_cooRowIndB,
		d_B_ColIndices, &pBufferSizeInBytesb);

	gpuErrchk(cudaMalloc(&pBufferb, sizeof(char)*pBufferSizeInBytesb));
	gpuErrchk(cudaMalloc(&Pb, sizeof(int)*nnzB));
	cusparseCreateIdentityPermutation(handle, nnzB, Pb);
	cusparseSafeCall(cusparseXcoosortByRow(handle, numBRows, numBColumns,
		nnzB,
		d_cooRowIndB,
		d_B_ColIndices,
		Pb,
		pBufferb));

	cusparseSafeCall(cusparseDgthr(handle, nnzB, d_B, d_B, Pb, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndB,
		nnzB,
		numBRows,
		d_B_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));

	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	int nnzC = 0;
	int baseC = 0;
// nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzC;
	//int *d_C_RowIndices;    gpuErrchk(cudaMalloc((void **)&d_C_RowIndices, sizeof(int)*(numARows + 1)));
	
   size_t pivot_dimensROW_C[1] = {numARows+1};
   
   mxGPUArray * ROW_C = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_C, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_C_RowIndices = (int *)mxGPUGetData(ROW_C);
	// where op ( A ) , op ( B ) and C are m×k (numARows×numAColumns), k×n(numBRows×numBColumns), and m×n(numARows×numBColumns) sparse matrices
	cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, numARows, numBColumns, numAColumns,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzB, d_B_RowIndices, d_B_ColIndices,
		descrA, d_C_RowIndices, nnzTotalDevHostPtr ));
       
        
        
     if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;
    }else{
    cudaMemcpy(&nnzC, d_C_RowIndices+numARows, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost);
    nnzC -= baseC;
     }

	gpuErrchk(cudaFree(pBufferb));
	gpuErrchk(cudaFree(Pb));
	//gpuErrchk(cudaFree(d_cooRowIndB));
	
    
   if (nnzC==0) {
          
         OUTPUTMATRIX = mxCreateSparse(numARows,numBColumns,0,mxREAL);
         
         return;
             
    } 
    
    
   size_t pivot_dimensionsrow[1] = {nnzC};
   size_t pivot_dimensionscolumn[1] = {numBColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnzC};
   mxGPUArray * ROW_SORTC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORTC = (int *)mxGPUGetData(ROW_SORTC);
   mxGPUArray * COL_SORTC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORTC = (int *)mxGPUGetData(COL_SORTC);
    mxGPUArray *VAL_SORTC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *VALSORTC = (double *)mxGPUGetData(VAL_SORTC);
   mwSize nnzm=(mwSize)nnzC;
   
   OUTPUTMATRIX = mxCreateSparse(numARows,numBColumns,nnzm,mxREAL);
    
    
   // double *d_C;          gpuErrchk(cudaMalloc((void **)&d_C, sizeof(double)*(nnzC)));
   // int *d_C_ColIndices;   gpuErrchk(cudaMalloc((void **)&d_C_ColIndices, sizeof(int)*(nnzC)));
	
	size_t pivot_dimensC[1] = {nnzC};
    size_t pivot_dimensCOL_C[1] = {nnzC};
   
   
   mxGPUArray *C = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensC, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_C = (double *)mxGPUGetData(C);
   mxGPUArray * COL_C = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_C, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_C_ColIndices = (int *)mxGPUGetData(COL_C);
    
    
	
	
	    
   	cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, numARows, numBColumns, numAColumns,
		descrA, nnzA,
		d_A, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzB,
		d_B, d_B_RowIndices, d_B_ColIndices,
		descrA,
		d_C, d_C_RowIndices, d_C_ColIndices));
        
	//gpuErrchk(cudaFree(d_A));
	//gpuErrchk(cudaFree(d_A_RowIndices));
	//gpuErrchk(cudaFree(d_A_ColIndices));
	
	//gpuErrchk(cudaFree(d_B));
	//gpuErrchk(cudaFree(d_B_RowIndices));
	//gpuErrchk(cudaFree(d_B_ColIndices));
	
	mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(ROW_A);
    mxGPUDestroyGPUArray(COL_A);
 
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(ROW_B);
    mxGPUDestroyGPUArray(COL_B);
    mxGPUDestroyGPUArray(COO_B);
	//double *d_value_csc;  gpuErrchk(cudaMalloc((void **)&d_value_csc, sizeof(double)*(nnzC)));
	//int *d_row_csc;       gpuErrchk(cudaMalloc((void **)&d_row_csc, sizeof(int)*(nnzC)));
	//int *d_col_csc;       gpuErrchk(cudaMalloc((void **)&d_col_csc, sizeof(int)*(numBColumns + 1)));

	cusparseSafeCall(cusparseDcsr2csc(handle, numARows, numBColumns, nnzC, d_C, d_C_RowIndices, d_C_ColIndices, VALSORTC, ROWSORTC, COLSORTC, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE));
    /*

	int *Pc = NULL;
	void *pBufferc = NULL;
	size_t pBufferSizeInBytesc = 0;
	cusparseXcscsort_bufferSizeExt(handle, numARows, numBColumns,
		nnzC,
		d_col_csc,
		d_row_csc, &pBufferSizeInBytesc);
   
	gpuErrchk(cudaMalloc(&pBufferc, sizeof(char)*pBufferSizeInBytesc));
	gpuErrchk(cudaMalloc(&Pc, sizeof(int)*nnzC));
	cusparseCreateIdentityPermutation(handle, nnzC, Pc);
	cusparseSafeCall(cusparseXcscsort(handle, numARows, numBColumns,
		nnzC,
		descrA,
		d_col_csc,
		d_row_csc,
		Pc,
		pBufferc));

	cusparseSafeCall(cusparseDgthr(handle, nnzC, d_value_csc, d_value_csc, Pc, CUSPARSE_INDEX_BASE_ZERO));
    */
  
	//gpuErrchk(cudaMemcpy(VALSORTC, d_value_csc, sizeof(double)* nnzC, cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(ROWSORTC, d_row_csc, sizeof(int)* (nnzC), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(COLSORTC, d_col_csc, sizeof(int)* (numBColumns + 1), cudaMemcpyDeviceToHost));

    //gpuErrchk(cudaFree(pBufferc));
	//gpuErrchk(cudaFree(Pc));
  
    
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORTC);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORTC);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  
        irs = static_cast<mwIndex *> (mxMalloc (nnzC * sizeof(mwIndex)));
                           int i;
	   #pragma omp parallel for shared(nnzC) private(i)
         for (i = 0; i < nnzC; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i])-1;  
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((numBColumns+1) * sizeof(mwIndex)));
      int nc1= numBColumns+1;
       #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i])-1;
            }
             
        mxDouble* PRS = (mxDouble*) mxMalloc (nnzC * sizeof(mxDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORTC, nnzC * sizeof(mxDouble), cudaMemcpyDeviceToHost));
		
           
    
   
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles(OUTPUTMATRIX)) ;
    
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s=mxSetDoubles(OUTPUTMATRIX, (mxDouble *)PRS);
         if ( s == 0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }    
    
	//gpuErrchk(cudaFree(d_C));
	//gpuErrchk(cudaFree(d_C_RowIndices));
	//gpuErrchk(cudaFree(d_C_ColIndices));

	//gpuErrchk(cudaFree(d_value_csc));
	//gpuErrchk(cudaFree(d_row_csc));
	//gpuErrchk(cudaFree(d_col_csc));

      mxGPUDestroyGPUArray(C);
      mxGPUDestroyGPUArray(ROW_C);
      mxGPUDestroyGPUArray(COL_C);



         mxDestroyArray(VLSXY2);
         mxGPUDestroyGPUArray(VAL_SORTC);
         mxGPUDestroyGPUArray(ROW_SORTC);
         mxGPUDestroyGPUArray(COL_SORTC);
         mxDestroyArray(RS);
         mxDestroyArray(CS);

         
         mxDestroyArray(COL_SORTB);
         mxDestroyArray(ROW_SORTB);
        cusparseDestroyMatDescr(descrA);  
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
           //     "Invalid input to MEX file, input(FIRST and SECOND  ARGUMENTS) must be  double precision.");
             
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
    if ( numAColumns != numBRows) {
	
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "Invalid input to MEX file, column number of dense matrix(first argument) must match the row number of sparse matrix(second argument).");
             
    }
    
   double *h_A_dense1;
   h_A_dense1 = (double *)mxGetDoubles(INPUTDENSEA);
    
    mwIndex nnz2;
 
    nnz2 = *(mxGetJc(INPUTSPARSEB) + numBColumns);
    int nnzB = (int)nnz2;
    
          
    mxArray *  ROW_SORTB = mxCreateNumericMatrix(nnzB, 1,mxINT32_CLASS, mxREAL);
    int *ROWSORTB  = (int *)mxGetInt32s(ROW_SORTB);
       SetIr_Data(INPUTSPARSEB, ROWSORTB);

    
   mxArray *  COL_SORTB = mxCreateNumericMatrix(nnzB, 1, mxINT32_CLASS, mxREAL);
    int  *COLSORTB = (int *)mxGetInt32s(COL_SORTB);
          SetJc_Int(INPUTSPARSEB, COLSORTB);

      
    double  *VALSORTB = (double *)mxGetDoubles(INPUTSPARSEB); 
          
     cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);        
    
    
    int nnzA = 0;                            // --- Number of nonzero elements in dense matrix A
	const int lda = numARows;
	//int *d_nnzPerVectorA;    gpuErrchk(cudaMalloc(&d_nnzPerVectorA, numARows * sizeof(*d_nnzPerVectorA)));
	size_t pivot_pervect[1] = {numARows};
    mxGPUArray *PerVect = mxGPUCreateGPUArray(1, (mwSize*) pivot_pervect, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	int *d_nnzPerVectorA = (int*)mxGPUGetData(PerVect);
    
    //double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, numARows * numAColumns * sizeof(*d_A_dense)));
	
	  size_t pivot_dimensionsvalueDA[2] = {numARows, numAColumns};
      mxGPUArray *OUTMA = mxGPUCreateGPUArray(2, (mwSize*) pivot_dimensionsvalueDA, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      double *d_A_dense = (double *)mxGPUGetData(OUTMA);
	
	
	
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense1, numARows * numAColumns * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, numARows, numAColumns, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));      
       // double *d_A;           // gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
		//int *d_A_RowIndices;   // gpuErrchk(cudaMalloc(&d_A_RowIndices, (numARows + 1) * sizeof(*d_A_RowIndices)));
		//int *d_A_ColIndices;   // gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
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
        //gpuErrchk(cudaFree(d_nnzPerVectorA));
          mxGPUDestroyGPUArray(PerVect);
	//double *d_B;           // gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));
	//int *d_B_RowIndices;   // gpuErrchk(cudaMalloc(&d_B_RowIndices, (numBRows + 1) * sizeof(*d_B_RowIndices)));
	//int *d_B_ColIndices;   // gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));
	//int *d_cooRowIndB;     //  gpuErrchk(cudaMalloc(&d_cooRowIndB, nnzB * sizeof(*d_cooRowIndB)));
	
   size_t pivot_dimensB[1] = {nnzB};
   size_t pivot_dimensROW_B[1] = {numBRows+1};
   size_t pivot_dimensCOL_B[1] = {nnzB};
   size_t pivot_dimensCOO_B[1] = {nnzB};
   
   mxGPUArray *B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensB, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_B = (double *)mxGPUGetData(B);
   mxGPUArray * ROW_B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_B, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_B_RowIndices = (int *)mxGPUGetData(ROW_B);
   mxGPUArray * COL_B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_B, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_B_ColIndices = (int *)mxGPUGetData(COL_B);
    mxGPUArray * COO_B = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOO_B, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_cooRowIndB = (int *)mxGPUGetData(COO_B);
    
    
    
    // --- Descriptor for sparse matrix B
	gpuErrchk(cudaMemcpy(d_B, VALSORTB, nnzB * sizeof(*d_B), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B_ColIndices, COLSORTB, nnzB * sizeof(*d_B_ColIndices), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cooRowIndB, ROWSORTB, nnzB * sizeof(*d_cooRowIndB), cudaMemcpyHostToDevice));
    
    
	int *Pb = NULL;
	void *pBufferb = NULL;
	size_t pBufferSizeInBytesb = 0;
	cusparseXcoosort_bufferSizeExt(handle, numBRows, numBColumns,
		nnzB,
		d_cooRowIndB,
		d_B_ColIndices, &pBufferSizeInBytesb);

	gpuErrchk(cudaMalloc(&pBufferb, sizeof(char)*pBufferSizeInBytesb));
	gpuErrchk(cudaMalloc(&Pb, sizeof(int)*nnzB));
	cusparseCreateIdentityPermutation(handle, nnzB, Pb);
	cusparseSafeCall(cusparseXcoosortByRow(handle, numBRows, numBColumns,
		nnzB,
		d_cooRowIndB,
		d_B_ColIndices,
		Pb,
		pBufferb));

	cusparseSafeCall(cusparseDgthr(handle, nnzB, d_B, d_B, Pb, CUSPARSE_INDEX_BASE_ZERO));

	cusparseSafeCall(cusparseXcoo2csr(handle,
		d_cooRowIndB,
		nnzB,
		numBRows,
		d_B_RowIndices,
		CUSPARSE_INDEX_BASE_ONE));

	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	int nnzC = 0;
	int baseC = 0;
// nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzC;
	//int *d_C_RowIndices;   // gpuErrchk(cudaMalloc((void **)&d_C_RowIndices, sizeof(int)*(numARows + 1)));
     size_t pivot_dimensROW_C[1] = {numARows+1};
   
    mxGPUArray * ROW_C = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensROW_C, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_C_RowIndices = (int *)mxGPUGetData(ROW_C);	
    
    // where op ( A ) , op ( B ) and C are m×k (numARows×numAColumns), k×n(numBRows×numBColumns), and m×n(numARows×numBColumns) sparse matrices
	cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, numARows, numBColumns, numAColumns,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzB, d_B_RowIndices, d_B_ColIndices,
		descrA, d_C_RowIndices, nnzTotalDevHostPtr ));
       
        
        
     if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;
    }else{
    cudaMemcpy(&nnzC, d_C_RowIndices+numARows, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost);
    nnzC -= baseC;
     }

	gpuErrchk(cudaFree(pBufferb));
	gpuErrchk(cudaFree(Pb));
	//gpuErrchk(cudaFree(d_cooRowIndB));
             
   
   if (nnzC==0) {
          
         OUTPUTMATRIX = mxCreateSparse(numARows,numBColumns,0,mxREAL);
         
         return;
             
    } 
    
      
  size_t pivot_dimensionsrow[1] = {nnzC};
   size_t pivot_dimensionscolumn[1] = {numBColumns+1}; 
   size_t pivot_dimensionsvalue[1] = {nnzC};
   mxGPUArray * ROW_SORTC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *ROWSORTC = (int *)mxGPUGetData(ROW_SORTC);
   mxGPUArray * COL_SORTC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionscolumn, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *COLSORTC = (int *)mxGPUGetData(COL_SORTC);
    mxGPUArray *VAL_SORTC = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *VALSORTC = (double *)mxGPUGetData(VAL_SORTC);
   mwSize nnzm=(mwSize)nnzC;
   
   OUTPUTMATRIX = mxCreateSparse(numARows,numBColumns,nnzm,mxREAL);
    
   // double *d_C;         // gpuErrchk(cudaMalloc((void **)&d_C, sizeof(double)*(nnzC)));
   // int *d_C_ColIndices; //  gpuErrchk(cudaMalloc((void **)&d_C_ColIndices, sizeof(int)*(nnzC)));
	   
   	size_t pivot_dimensC[1] = {nnzC};
    size_t pivot_dimensCOL_C[1] = {nnzC};
   
   
   mxGPUArray *C = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensC, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    double  *d_C = (double *)mxGPUGetData(C);
   mxGPUArray * COL_C = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensCOL_C, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int  *d_C_ColIndices = (int *)mxGPUGetData(COL_C); 
    
   	cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, numARows, numBColumns, numAColumns,
		descrA, nnzA,
		d_A, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzB,
		d_B, d_B_RowIndices, d_B_ColIndices,
		descrA,
		d_C, d_C_RowIndices, d_C_ColIndices));
        
	//gpuErrchk(cudaFree(d_A));
	//gpuErrchk(cudaFree(d_A_RowIndices));
	//gpuErrchk(cudaFree(d_A_ColIndices));
	
	//gpuErrchk(cudaFree(d_B));
	//gpuErrchk(cudaFree(d_B_RowIndices));
	//gpuErrchk(cudaFree(d_B_ColIndices));
    
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(ROW_A);
    mxGPUDestroyGPUArray(COL_A);
 
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(ROW_B);
    mxGPUDestroyGPUArray(COL_B);
    mxGPUDestroyGPUArray(COO_B);
    
    
	cusparseSafeCall(cusparseDcsr2csc(handle, numARows, numBColumns, nnzC, d_C, d_C_RowIndices, d_C_ColIndices, VALSORTC, ROWSORTC, COLSORTC, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE));
    /*

	int *Pc = NULL;
	void *pBufferc = NULL;
	size_t pBufferSizeInBytesc = 0;
	cusparseXcscsort_bufferSizeExt(handle, numARows, numBColumns,
		nnzC,
		d_col_csc,
		d_row_csc, &pBufferSizeInBytesc);
   
	gpuErrchk(cudaMalloc(&pBufferc, sizeof(char)*pBufferSizeInBytesc));
	gpuErrchk(cudaMalloc(&Pc, sizeof(int)*nnzC));
	cusparseCreateIdentityPermutation(handle, nnzC, Pc);
	cusparseSafeCall(cusparseXcscsort(handle, numARows, numBColumns,
		nnzC,
		descrA,
		d_col_csc,
		d_row_csc,
		Pc,
		pBufferc));

	cusparseSafeCall(cusparseDgthr(handle, nnzC, d_value_csc, d_value_csc, Pc, CUSPARSE_INDEX_BASE_ZERO));
    */
  
	//gpuErrchk(cudaMemcpy(VALSORTC, d_value_csc, sizeof(double)* nnzC, cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(ROWSORTC, d_row_csc, sizeof(int)* (nnzC), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(COLSORTC, d_col_csc, sizeof(int)* (numBColumns + 1), cudaMemcpyDeviceToHost));

    //gpuErrchk(cudaFree(pBufferc));
	//gpuErrchk(cudaFree(Pc));
  
    
   mxArray *RS= mxGPUCreateMxArrayOnCPU(ROW_SORTC);
   int * rs= (int *)mxGetInt32s(RS);
   mxArray *CS= mxGPUCreateMxArrayOnCPU(COL_SORTC);
   int * cs= (int *)mxGetInt32s(CS);

    
      mwIndex *irs,*jcs;
  
        irs = static_cast<mwIndex *> (mxMalloc (nnzC * sizeof(mwIndex)));
                           int i;
	   #pragma omp parallel for shared(nnzC) private(i)
         for (i = 0; i < nnzC; ++i) {
           irs[i] = static_cast<mwIndex> (rs[i])-1;  
            }
      
      jcs = static_cast<mwIndex *> (mxMalloc ((numBColumns+1) * sizeof(mwIndex)));
      int nc1= numBColumns+1;
       #pragma omp parallel for shared(nc1) private(i)
            for (i = 0; i < nc1; ++i) {
           jcs[i] = static_cast<mwIndex> (cs[i])-1;
            }
             
        mxDouble* PRS = (mxDouble*) mxMalloc (nnzC * sizeof(mxDouble));
        gpuErrchk(cudaMemcpy(PRS, VALSORTC, nnzC * sizeof(mxDouble), cudaMemcpyDeviceToHost));          

           
    
   
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetDoubles(OUTPUTMATRIX)) ;
    
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s=mxSetDoubles(OUTPUTMATRIX, (mxDouble *)PRS);
         if ( s == 0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         } 
        

	//gpuErrchk(cudaFree(d_C));
	//gpuErrchk(cudaFree(d_C_RowIndices));
	//gpuErrchk(cudaFree(d_C_ColIndices));
      mxGPUDestroyGPUArray(C);
      mxGPUDestroyGPUArray(ROW_C);
      mxGPUDestroyGPUArray(COL_C);
      
	//gpuErrchk(cudaFree(d_value_csc));
	//gpuErrchk(cudaFree(d_row_csc));
	//gpuErrchk(cudaFree(d_col_csc));

         mxGPUDestroyGPUArray(VAL_SORTC);
         mxGPUDestroyGPUArray(ROW_SORTC);
         mxGPUDestroyGPUArray(COL_SORTC);
         mxDestroyArray(RS);
         mxDestroyArray(CS);

         
         mxDestroyArray(COL_SORTB);
         mxDestroyArray(ROW_SORTB);
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
