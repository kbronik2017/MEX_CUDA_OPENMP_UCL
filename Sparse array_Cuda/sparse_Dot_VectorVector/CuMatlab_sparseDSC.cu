
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
    if ( (((numARows!= 1) && (numAColumns!= 1))) ||(((numBRows!= 1) && (numBColumns!= 1)))) {
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, first/second arguments must be dense/sparse column/row vector.");
             
    }
    
    if ( mxGPUGetNumberOfElements(INPUTDENSEGPUA)!=mxGPUGetNumberOfElements(INPUTSPARSEGPUB)) {
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB);    
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, row/column number of dense vector(first argument) must be equal to row/column numbers of sparse vector(second argument).");
             
    }
 
      
  
    mwIndex nnz2;
    mxArray * tempx = mxGPUCreateMxArrayOnCPU(INPUTSPARSEGPUB);
    nnz2 = *(mxGetJc(tempx) + numBColumns);
    int nnz = (int)nnz2;
    
    int *pointerrow =0;
    mxArray *row_sort;
   if (numBColumns == 1) {
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
   
    Ir_DataGetSetIXY(tempx , pointerrow, nnz);
    }
    
   if (numBRows == 1) {

   
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
    
    Jc_GetSetIXY(tempx , pointerrow);
   
        } 
   
    cuDoubleComplex  *pointerval = (cuDoubleComplex *)mxGetComplexDoubles(tempx);
            
   size_t pivot_dimensionsrow[1] = {nnz};
   
   size_t pivot_dimensionsvalue[1] = {nnz};    
      mxGPUArray *row_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortA=(int *)mxGPUGetData(row_sortA);
 gpuErrchk(cudaMemcpy(xrow_sortA, pointerrow, nnz * sizeof(*xrow_sortA), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortA=(cuDoubleComplex*)mxGPUGetData(val_sortA);
 gpuErrchk(cudaMemcpy(xval_sortA, pointerval, nnz * sizeof(*xval_sortA), cudaMemcpyHostToDevice));    
   
       
     cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);        
    
   cuDoubleComplex const *d_A_dense;
   d_A_dense = (cuDoubleComplex const *)(mxGPUGetDataReadOnly(INPUTDENSEGPUA));  
    
              mxGPUDestroyGPUArray(INPUTDENSEGPUA);
              mxGPUDestroyGPUArray(INPUTSPARSEGPUB); 
              mxDestroyArray(row_sort);
              mxDestroyArray(tempx); 
              
   cuDoubleComplex VALOUT= make_cuDoubleComplex(0.0, 0.0); 
  cusparseSafeCall(cusparseZdotci(handle, nnz, 
              xval_sortA, 
              xrow_sortA, d_A_dense, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));              
  
      int nnzx=1;    
      mwSize nnzm=(mwSize)nnzx;
      OUTPUTMATRIX = mxCreateSparse(1,1,nnzm,mxCOMPLEX);
  
      mwIndex *irs = static_cast<mwIndex *> (mxMalloc (nnzx * sizeof(mwIndex)));
      irs[0] = 0;

      mwIndex *jcs = static_cast<mwIndex *> (mxMalloc (2 * sizeof(mwIndex)));
      jcs[0]=jcs[1]=1;
        
      mxComplexDouble* PRS =static_cast<mxComplexDouble *> (mxMalloc (nnzx * sizeof(mxComplexDouble)));

		 PRS[0].real = VALOUT.x;
         PRS[0].imag = (-1)*(VALOUT.y);
		 
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetComplexDoubles (OUTPUTMATRIX)) ;      
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s = mxSetComplexDoubles(OUTPUTMATRIX, (mxComplexDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
        mxGPUDestroyGPUArray(row_sortA);
        mxGPUDestroyGPUArray(val_sortA);
 
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
    if ( (((numARows!= 1) && (numAColumns!= 1))) ||(((numBRows!= 1) && (numBColumns!= 1)))) {
   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, first/second arguments must be dense/sparse column/row vector.");
             
    }
    
     if ( mxGetNumberOfElements(INPUTDENSEA)!=mxGetNumberOfElements(INPUTSPARSEB)) {
   
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                        "Invalid input to MEX file, row/column number of dense vector(first argument) must be equal to row/column numbers of sparse vector(second argument).");
             
    }
    
      
   
    
	mwIndex nnz2;
 
    nnz2 = *(mxGetJc(INPUTSPARSEB) + numBColumns);
    int nnz = (int)nnz2;
    
    int *pointerrow =0;
    mxArray *row_sort;
   if (numBColumns == 1) {
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
   
    Ir_DataGetSetIXY(INPUTSPARSEB , pointerrow, nnz);
    
    }
    
   if (numBRows == 1) {

   
    row_sort =mxCreateNumericMatrix(nnz, 1, mxINT32_CLASS, mxREAL);
    pointerrow = (int *)mxGetInt32s(row_sort);
    
    Jc_GetSetIXY(INPUTSPARSEB , pointerrow);
   
        }
   
    cuDoubleComplex  *pointerval = (cuDoubleComplex *)mxGetComplexDoubles(INPUTSPARSEB);
            
   size_t pivot_dimensionsrow[1] = {nnz};
   
   size_t pivot_dimensionsvalue[1] = {nnz};    
      mxGPUArray *row_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsrow, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
          
        int *xrow_sortA=(int *)mxGPUGetData(row_sortA);
       gpuErrchk(cudaMemcpy(xrow_sortA, pointerrow, nnz * sizeof(*xrow_sortA), cudaMemcpyHostToDevice));
       
      mxGPUArray *val_sortA = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalue, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);  
          
        cuDoubleComplex *xval_sortA=(cuDoubleComplex*)mxGPUGetData(val_sortA);
      gpuErrchk(cudaMemcpy(xval_sortA, pointerval, nnz * sizeof(*xval_sortA), cudaMemcpyHostToDevice)); 
 
          
     cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);        
    
  
       cuDoubleComplex *h_A_dense1;
       h_A_dense1 = (cuDoubleComplex *)mxGetComplexDoubles(INPUTDENSEA);
 
              mxDestroyArray(row_sort);
               
      
   cuDoubleComplex  *VALDENSE;
   mxGPUArray *VAL;
if (numAColumns == 1) {	
    
  size_t   pivot_dimensionsvalueV[1] = {numARows};
  VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    VALDENSE = (cuDoubleComplex *)mxGPUGetData(VAL);
       gpuErrchk(cudaMemcpy(VALDENSE, h_A_dense1, sizeof(cuDoubleComplex) * numARows , cudaMemcpyHostToDevice));
    }  

if (numARows == 1) {
	
   size_t  pivot_dimensionsvalueV[1] = {numAColumns};
   VAL = mxGPUCreateGPUArray(1, (mwSize*) pivot_dimensionsvalueV, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    VALDENSE = (cuDoubleComplex *)mxGPUGetData(VAL);
       gpuErrchk(cudaMemcpy(VALDENSE, h_A_dense1, sizeof(cuDoubleComplex) * numAColumns , cudaMemcpyHostToDevice));
   
}   
		
	
   cuDoubleComplex VALOUT= make_cuDoubleComplex(0.0, 0.0); 
  cusparseSafeCall(cusparseZdotci(handle, nnz, 
              xval_sortA, 
              xrow_sortA, VALDENSE, 
              &VALOUT, 
              CUSPARSE_INDEX_BASE_ONE));              
  
      int nnzx=1;    
      mwSize nnzm=(mwSize)nnzx;
      OUTPUTMATRIX = mxCreateSparse(1,1,nnzm,mxCOMPLEX);
  
      mwIndex *irs = static_cast<mwIndex *> (mxMalloc (nnzx * sizeof(mwIndex)));
      irs[0] = 0;

      mwIndex *jcs = static_cast<mwIndex *> (mxMalloc (2 * sizeof(mwIndex)));
      jcs[0]=jcs[1]=1;
        
      mxComplexDouble* PRS =static_cast<mxComplexDouble *> (mxMalloc (nnzx * sizeof(mxComplexDouble)));
         PRS[0].real = VALOUT.x;
         PRS[0].imag = (-1)*(VALOUT.y);
		 
        mxFree (mxGetJc (OUTPUTMATRIX)) ;
        mxFree (mxGetIr (OUTPUTMATRIX)) ;
        mxFree (mxGetComplexDoubles (OUTPUTMATRIX)) ;      
        mxSetIr(OUTPUTMATRIX, (mwIndex *)irs);
        mxSetJc(OUTPUTMATRIX, (mwIndex *)jcs);
        int s = mxSetComplexDoubles(OUTPUTMATRIX, (mxComplexDouble *)PRS);
        if ( s==0) {
         mexErrMsgIdAndTxt( "MATLAB:mexatexit:invalidInput",
                "the function is unsuccessful, either mxArray is not an unshared mxDOUBLE_CLASS array, or the data is not allocated with mxCalloc.");
             
         }
        mxGPUDestroyGPUArray(row_sortA);
        mxGPUDestroyGPUArray(val_sortA);
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
